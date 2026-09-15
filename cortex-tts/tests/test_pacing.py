"""The planner, the render model and the sentence buffer, driven without a clock.

Every figure the planner uses comes from a `RenderModel` built here to look
like a measured host: a fast whole-render engine (the 40M on a GPU), a
whole-render engine at real time with a fixed cost (OmniVoice with a clone), a
chunk-streaming engine slower than real time (MOSS on a CPU). What is pinned
is which of the three ways to speak a reply each one gets, and that the
opening request has content.
"""

from __future__ import annotations

import pytest

from cortex_speech.pacing import (
    BUFFERED,
    PACED,
    STREAMING,
    Finished,
    Planner,
    RenderModel,
    RenderSample,
    Send,
    SentenceBuffer,
    Wait,
    clause_pieces,
)
from cortex_speech.pacing.planner import HOLD_CAP_S

# Chinese at 4 chars/s so the arithmetic below is legible: 12 chars = 3 s.
FAST_WHOLE = RenderModel(
    fixed_s=0.3, per_audio=0.4, cjk_per_s=4.0, latin_per_s=14.0, samples=12
)
REALTIME_WHOLE = RenderModel(
    fixed_s=1.5, per_audio=1.0, cjk_per_s=4.0, latin_per_s=14.0, samples=12
)
SLOW_CHUNKED = RenderModel(
    fixed_s=0.3, per_audio=1.15, cjk_per_s=4.0, latin_per_s=14.0, samples=12
)
FAST_CHUNKED = RenderModel(
    fixed_s=0.3, per_audio=0.4, cjk_per_s=4.0, latin_per_s=14.0, samples=12
)

S1 = "從前有一座山，山上有一間小廟。"  # 15 chars ≈ 3.75 s
S2 = "廟裡住著一位老和尚和一位小和尚。"  # 16 chars ≈ 4 s
S3 = "每天早上他們都到溪邊打水。"  # 13 chars ≈ 3.25 s
S4 = "日子過得很平靜。"  # 8 chars ≈ 2 s


class TestSentenceBuffer:
    def test_a_sentence_is_complete_only_once_its_stop_arrives(self) -> None:
        buffer = SentenceBuffer()
        buffer.feed("好了，燈")
        assert buffer.sentences == []
        assert buffer.tail == "好了，燈"
        buffer.feed("已經打開了。")
        assert buffer.sentences == ["好了，燈已經打開了。"]
        assert buffer.tail == ""

    def test_the_tail_waits_while_the_sentences_can_be_taken(self) -> None:
        buffer = SentenceBuffer()
        buffer.feed(S1 + S2 + "日子")
        assert buffer.take_sentences(1) == [S1]
        assert buffer.sentences == [S2]
        assert buffer.tail == "日子"

    def test_an_ascii_stop_needs_the_space_after_it(self) -> None:
        """`3.` may be `3.5`; only `3. ` is a sentence end."""
        buffer = SentenceBuffer()
        buffer.feed("It is 26.")
        assert buffer.sentences == []
        buffer.feed("5 degrees. Fine")
        assert buffer.sentences == ["It is 26.5 degrees."]

    def test_ending_makes_the_tail_a_sentence(self) -> None:
        buffer = SentenceBuffer()
        buffer.feed("好的")
        buffer.end()
        assert buffer.sentences == ["好的"]
        assert buffer.tail == ""

    def test_take_prefix_removes_exactly_the_cut(self) -> None:
        buffer = SentenceBuffer()
        buffer.feed("從前有一座山，山上有一間小廟")
        buffer.take_prefix("從前有一座山，")
        assert buffer.tail == "山上有一間小廟"

    def test_clause_pieces_keep_the_mark_with_the_text_before_it(self) -> None:
        assert clause_pieces("一，二、三；四") == ["一，", "二、", "三；", "四"]


class TestRenderModelFit:
    @staticmethod
    def _samples(
        fixed: float, factor: float, lengths: list[float]
    ) -> list[RenderSample]:
        return [
            RenderSample(audio_s=a, wall_s=fixed + factor * a, cjk=int(a * 4), latin=0)
            for a in lengths
        ]

    def test_recovers_the_fixed_cost_and_the_factor(self) -> None:
        model = RenderModel.fit(self._samples(1.5, 0.95, [2, 4, 6, 9, 12]))
        assert model is not None
        assert model.fixed_s == pytest.approx(1.5, abs=0.05)
        assert model.per_audio == pytest.approx(0.95, abs=0.01)
        assert model.cjk_per_s == pytest.approx(4.0, abs=0.1)

    def test_too_few_samples_is_no_model(self) -> None:
        assert RenderModel.fit(self._samples(1.0, 1.0, [3, 4])) is None

    def test_equal_lengths_fall_back_to_the_ratio(self) -> None:
        """No slope to fit: the factor is the median RTF, the fixed part zero."""
        model = RenderModel.fit(self._samples(0.0, 0.7, [5, 5.1, 5.2, 5.0]))
        assert model is not None
        assert model.fixed_s == 0.0
        assert model.per_audio == pytest.approx(0.7, abs=0.01)

    def test_a_negative_intercept_is_clamped_not_believed(self) -> None:
        noisy = self._samples(0.0, 0.5, [2, 4, 8])
        noisy[0] = RenderSample(audio_s=2, wall_s=0.4, cjk=8, latin=0)
        model = RenderModel.fit(noisy)
        assert model is not None
        assert model.fixed_s == 0.0
        assert model.per_audio > 0

    def test_a_script_never_spoken_keeps_its_prior(self) -> None:
        model = RenderModel.fit(self._samples(0.0, 0.5, [2, 4, 8]))
        assert model is not None
        assert model.latin_per_s == 14.7

    def test_deficit_of_a_whole_render_is_the_whole_render(self) -> None:
        assert REALTIME_WHOLE.deficit(4.0, chunk_streaming=False) == pytest.approx(5.5)

    def test_deficit_of_a_chunked_render_is_prefill_plus_drain(self) -> None:
        assert SLOW_CHUNKED.deficit(10.0, chunk_streaming=True) == pytest.approx(1.8)
        assert FAST_CHUNKED.deficit(10.0, chunk_streaming=True) == pytest.approx(0.3)


class TestUnmeasuredModelIsBuffered:
    def test_nothing_goes_out_before_the_end(self) -> None:
        planner = Planner(None, chunk_streaming=False)
        planner.feed(S1 + S2)
        assert planner.mode == BUFFERED
        assert isinstance(planner.plan(None), Wait)
        planner.end()
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.text == S1 + S2
        assert decision.hold_all
        assert isinstance(planner.plan(0.0), Finished)

    def test_asking_for_buffered_overrides_a_measured_model(self) -> None:
        planner = Planner(FAST_WHOLE, chunk_streaming=False, mode=BUFFERED)
        assert planner.mode == BUFFERED


class TestTheFirstRequest:
    def test_a_bare_opener_waits_for_content(self) -> None:
        """`好了，` alone is under the floor, comma or not."""
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        planner.feed("好了，")
        assert isinstance(planner.plan(None), Wait)

    def test_a_sentence_past_the_floor_goes_out(self) -> None:
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        planner.feed("好了，" + S1)
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.text == "好了，" + S1
        assert planner.mode == STREAMING

    def test_short_sentences_are_gathered_up_to_the_floor(self) -> None:
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        planner.feed("好了。")  # 3 chars, well under
        assert isinstance(planner.plan(None), Wait)
        planner.feed(S4)  # 2 s more: still under 3 s
        assert isinstance(planner.plan(None), Wait)
        planner.feed(S3)
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.text == "好了。" + S4 + S3

    def test_the_first_batch_stops_at_the_ceiling(self) -> None:
        """A fast writer must not turn the opening into the whole reply."""
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        planner.feed(S1 + S2 + S3 + S4)
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.text == S1  # S1 + S2 would be 7.75 s, over 6

    def test_a_run_on_first_sentence_is_cut_where_the_rest_lands_in_time(self) -> None:
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        # 28 chars = 7 s and no stop yet: past the ceiling.
        planner.feed("從前有一座山，山上有一間小廟，廟裡住著一位老和尚和一位小和尚")
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.text == "從前有一座山，山上有一間小廟，"

    def test_a_run_on_first_sentence_waits_when_the_rest_would_not(self) -> None:
        """Cutting would only put a pause mid-sentence; whole is better."""
        planner = Planner(REALTIME_WHOLE, chunk_streaming=False)
        planner.feed("從前有一座山，山上有一間小廟，廟裡住著一位老和尚和一位小和尚")
        assert isinstance(planner.plan(None), Wait)

    def test_a_reply_finished_before_the_first_send_is_paced(self) -> None:
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        planner.feed(S4)
        planner.end()
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.text == S4
        assert decision.hold_bank
        assert planner.mode == PACED
        assert isinstance(planner.plan(0.0), Finished)


class TestStreaming:
    def test_the_next_batch_is_the_largest_the_lead_covers(self) -> None:
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        planner.feed(S1)
        assert isinstance(planner.plan(None), Send)
        planner.feed(S2 + S3 + S4)
        # Lead 10 s: S2+S3+S4 (9.25 s) renders in 0.3 + 3.7 = 4 s, fits.
        decision = planner.plan(10.0)
        assert isinstance(decision, Send)
        assert decision.text == S2 + S3 + S4

    def test_a_thin_lead_gets_one_sentence_rather_than_a_wait(self) -> None:
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        planner.feed(S1)
        planner.plan(None)
        planner.feed(S2 + S3)
        decision = planner.plan(0.2)
        assert isinstance(decision, Send)
        assert decision.text == S2

    def test_waits_for_text_and_finishes_after_the_end(self) -> None:
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        planner.feed(S1)
        planner.plan(None)
        assert isinstance(planner.plan(3.0), Wait)
        planner.feed("完")
        planner.end()
        decision = planner.plan(3.0)
        assert isinstance(decision, Send)
        assert decision.text == "完"
        assert isinstance(planner.plan(3.0), Finished)

    def test_a_whole_render_engine_gets_a_small_wall_hold(self) -> None:
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        planner.feed(S1)
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.hold_audio_s == 0.0
        assert 0 < decision.hold_wall_s <= 1.0

    def test_a_chunk_streaming_engine_banks_audio_instead(self) -> None:
        planner = Planner(FAST_CHUNKED, chunk_streaming=True)
        planner.feed(S1)
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.hold_wall_s == 0.0
        assert decision.hold_audio_s == pytest.approx(0.3 + 0.5)


class TestModelsThatCannotGainLead:
    """OmniVoice with a clone, MOSS on a CPU: paced, never streamed."""

    @pytest.mark.parametrize(
        ("model", "chunked"), [(REALTIME_WHOLE, False), (SLOW_CHUNKED, True)]
    )
    def test_waits_for_the_whole_reply(self, model: RenderModel, chunked: bool) -> None:
        planner = Planner(model, chunk_streaming=chunked)
        planner.feed(S1 + S2)
        assert isinstance(planner.plan(None), Wait)
        assert planner.mode == PACED
        planner.feed(S3 + S4)
        assert isinstance(planner.plan(None), Wait)
        planner.end()
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.text.startswith(S1)
        assert not decision.hold_all

    def test_the_paced_hold_is_exact_for_a_whole_render_engine(self) -> None:
        """Six 4 s sentences at RTF 1 with 1.5 s fixed, grouped three and three:
        the second batch lands 1.5 s after the first has played out, so the
        first is held that long plus the margin."""
        planner = Planner(REALTIME_WHOLE, chunk_streaming=False)
        planner.feed(S2 * 6)  # 96 chars = 24 s
        planner.end()
        first = planner.plan(None)
        assert isinstance(first, Send)
        batches = [first.text]
        while isinstance(nxt := planner.plan(0.0), Send):
            batches.append(nxt.text)
        assert batches == [S2 * 3, S2 * 3]
        assert first.hold_bank

    def test_a_second_batch_that_lands_in_time_needs_only_the_margin(self) -> None:
        planner = Planner(REALTIME_WHOLE, chunk_streaming=False)
        planner.feed(S1 + S2 + S3 + S2)  # groups as 11 s + 4 s
        planner.end()
        first = planner.plan(None)
        assert isinstance(first, Send)
        assert first.text == S1 + S2 + S3
        assert first.hold_bank

    def test_a_hold_past_the_cap_is_paced_even_when_lead_is_gained(self) -> None:
        heavy = RenderModel(
            fixed_s=HOLD_CAP_S + 1,
            per_audio=0.2,
            cjk_per_s=4.0,
            latin_per_s=14.0,
            samples=9,
        )
        planner = Planner(heavy, chunk_streaming=True)
        planner.feed(S1)
        assert isinstance(planner.plan(None), Wait)
        assert planner.mode == PACED


class TestTextArrivingACharacterAtATime:
    """What a writer actually looks like: the planner is asked after every
    delta, so its first decision often falls on a clause cut."""

    @staticmethod
    def _drive(planner: Planner, text: str, step: int = 2) -> list[str]:
        sent: list[str] = []
        for i in range(0, len(text), step):
            planner.feed(text[i : i + step])
            if i + step >= len(text):
                planner.end()
            decision = planner.plan(None if not sent else 3.0)
            if isinstance(decision, Send):
                sent.append(decision.text)
        while isinstance(decision := planner.plan(3.0), Send):
            sent.append(decision.text)
        assert isinstance(decision, Finished)
        return sent

    def test_a_lead_gaining_model_streams_and_nothing_is_lost(self) -> None:
        text = (S1 + S2) * 5
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        sent = self._drive(planner, text)
        assert planner.mode == STREAMING
        assert "".join(sent) == text

    def test_a_model_that_cannot_gain_lead_paces_and_nothing_is_lost(self) -> None:
        text = (S1 + S2) * 5
        planner = Planner(REALTIME_WHOLE, chunk_streaming=False)
        sent = self._drive(planner, text)
        assert planner.mode == PACED
        assert "".join(sent) == text

    def test_the_verdict_does_not_depend_on_where_the_first_cut_fell(self) -> None:
        """A short first piece must not make a fast model look slow."""
        barely = RenderModel(
            fixed_s=0.3, per_audio=0.8, cjk_per_s=4.0, latin_per_s=14.0, samples=12
        )
        run_on = "從前有一座山，山上有一間小廟，廟裡住著一位老和尚和一位小和尚。"
        planner = Planner(barely, chunk_streaming=False)
        sent = self._drive(planner, run_on * 4)
        assert planner.mode == STREAMING
        assert "".join(sent) == run_on * 4


class TestTheFitLeansSlow:
    def test_noisy_requests_raise_the_fixed_cost_by_their_spread(self) -> None:
        """Two hosts with the same average: the jittery one must predict slower."""
        clean = [
            RenderSample(audio_s=a, wall_s=0.3 + 0.8 * a, cjk=int(a * 4), latin=0)
            for a in (2, 4, 6, 8, 10, 12)
        ]
        jitter = [0.6, -0.6, 0.6, -0.6, 0.6, -0.6]
        noisy = [
            RenderSample(audio_s=s.audio_s, wall_s=s.wall_s + j, cjk=s.cjk, latin=0)
            for s, j in zip(clean, jitter, strict=True)
        ]
        steady, shaky = RenderModel.fit(clean), RenderModel.fit(noisy)
        assert steady is not None and shaky is not None
        assert steady.fixed_s == pytest.approx(0.3, abs=0.02)
        assert steady.spread_s == pytest.approx(0.0, abs=0.01)
        assert shaky.spread_s > 0.4
        assert shaky.fixed_s == pytest.approx(steady.fixed_s, abs=0.7)
        assert shaky.per_audio == pytest.approx(steady.per_audio, abs=0.1)

    def test_the_spread_widens_a_hold_once_not_per_request(self) -> None:
        """Eleven requests on a jittery host: the hold grows by one spread."""
        steady = RenderModel(
            fixed_s=0.3, per_audio=1.1, cjk_per_s=4.0, latin_per_s=14.0, samples=12
        )
        shaky = RenderModel(
            fixed_s=0.3,
            per_audio=1.1,
            cjk_per_s=4.0,
            latin_per_s=14.0,
            samples=12,
            spread_s=1.5,
        )
        holds = []
        for model in (steady, shaky):
            planner = Planner(model, chunk_streaming=True)
            planner.feed(S2 * 11)
            planner.end()
            first = planner.plan(None)
            assert isinstance(first, Send)
            holds.append(planner.bank_needed(0.0))
        assert holds[1] - holds[0] == pytest.approx(1.5, abs=0.05)


class TestBankNeeded:
    """A paced reply releases when the bank covers what is still to be lost."""

    def test_a_reply_that_runs_dear_widens_its_own_hold(self) -> None:
        """The fit is the host's average day; this reply may not get one.

        Measured in production: MOSS rendered a 61 s reply at 1.70x where its
        fit said 1.09x, and the hold sized from the fit let the listener run
        35 s dry. The first request had already been through the slow patch
        before the hold released, so the reply knew and the plan did not.
        """
        planner = Planner(SLOW_CHUNKED, chunk_streaming=True)
        planner.feed(S2 * 6)
        planner.end()
        assert isinstance(planner.plan(None), Send)
        unaware = planner.bank_needed(0.0)

        # The first request cost twice what the fit predicted for its audio.
        predicted = SLOW_CHUNKED.render_seconds(12.0)
        planner.rendered(12.0, predicted * 2)
        assert planner.bank_needed(0.0) > unaware

        # A batch that has produced almost nothing is not evidence: the
        # prediction tends to zero at the top of a request while the wall does
        # not, and on a host with no fixed cost the ratio ran away.
        bare = Planner(SLOW_CHUNKED, chunk_streaming=True)
        bare.feed(S2 * 6)
        bare.end()
        assert isinstance(bare.plan(None), Send)
        assert bare.bank_needed(0.001, 5.0) == pytest.approx(
            bare.bank_needed(0.0, 0.0), abs=0.01
        )

        # Past that, it counts before the request ends, because the hold is
        # normally released part-way through it.
        fresh = Planner(SLOW_CHUNKED, chunk_streaming=True)
        fresh.feed(S2 * 6)
        fresh.end()
        assert isinstance(fresh.plan(None), Send)
        half = SLOW_CHUNKED.render_seconds(6.0)
        assert fresh.bank_needed(6.0, half * 2) > fresh.bank_needed(6.0, half)

        # The whole line is dearer, and the loss comes off the moved line:
        # at a fitted 1.15 running 2x dear the engine loses 1.3 per audio
        # second, not (1.15 - 1) x 2 = 0.3.
        dear = SLOW_CHUNKED.scaled(2.0)
        assert dear.per_audio == pytest.approx(SLOW_CHUNKED.per_audio * 2)
        # The request already rendering has paid its fixed part, so only the
        # one behind it carries one.
        in_flight = max(0.0, dear.per_audio - 1.0) * 12.0
        queued = dear.deficit(12.0, chunk_streaming=True)
        assert planner.bank_needed(0.0) == pytest.approx(
            in_flight + queued + 0.5 + dear.spread_s, abs=0.1
        )

    def test_a_line_is_scaled_before_the_loss_is_taken_from_it(self) -> None:
        """Scaling the difference under-corrects, which is how it read -3.4.

        Measured against a host made 1.4x slower than its fit: scaling the
        deficit left the listener 3.4 s short where scaling the line covers it.
        """
        fit = SLOW_CHUNKED
        moved = fit.scaled(1.1)
        assert moved.per_audio - 1.0 > (fit.per_audio - 1.0) * 1.1
        # A factor at or below 1 is the fit itself: a hold never shrinks on
        # one cheap request.
        assert fit.scaled(1.0) is fit
        assert fit.scaled(0.5) is fit

    def test_a_reply_that_runs_cheap_does_not_narrow_it(self) -> None:
        """A hold too long is a wait; too short is a gap nobody can un-hear."""
        planner = Planner(SLOW_CHUNKED, chunk_streaming=True)
        planner.feed(S2 * 6)
        planner.end()
        assert isinstance(planner.plan(None), Send)
        unaware = planner.bank_needed(0.0)
        planner.rendered(12.0, SLOW_CHUNKED.render_seconds(12.0) / 4)
        assert planner.bank_needed(0.0) == pytest.approx(unaware, abs=0.01)

    def test_a_chunked_engine_counts_what_the_rest_will_lose(self) -> None:
        planner = Planner(SLOW_CHUNKED, chunk_streaming=True)
        planner.feed(S2 * 6)  # two batches of three sentences, 12 s each
        planner.end()
        first = planner.plan(None)
        assert isinstance(first, Send)
        assert first.hold_bank
        # Nothing produced yet: this batch loses 0.15 x 12, the next 0.3 + 0.15 x 12.
        assert planner.bank_needed(0.0) == pytest.approx(1.8 + 2.1 + 0.5, abs=0.05)
        # Half of this batch produced: its remaining loss halves.
        assert planner.bank_needed(6.0) == pytest.approx(0.9 + 2.1 + 0.5, abs=0.05)
        # On to the last batch, nothing produced: only its own loss.
        assert isinstance(planner.plan(0.0), Send)
        assert planner.bank_needed(0.0) == pytest.approx(1.8 + 0.5, abs=0.05)
        assert isinstance(planner.plan(0.0), Finished)
        assert planner.bank_needed(0.0) == 0.0

    def test_a_whole_engine_counts_the_deepest_point(self) -> None:
        planner = Planner(REALTIME_WHOLE, chunk_streaming=False)
        planner.feed(S2 * 6)
        planner.end()
        first = planner.plan(None)
        assert isinstance(first, Send)
        # Batch 1 renders in 13.5 s and lands whole; batch 2 lands at 27 s
        # against 12 s played: 15 s is the deepest point.
        assert planner.bank_needed(0.0) == pytest.approx(15.0 + 0.5, abs=0.05)

    def test_the_spread_is_charged_once(self) -> None:
        shaky = RenderModel(
            fixed_s=0.3,
            per_audio=1.15,
            cjk_per_s=4.0,
            latin_per_s=14.0,
            samples=12,
            spread_s=1.0,
        )
        planner = Planner(shaky, chunk_streaming=True)
        planner.feed(S2 * 6)
        planner.end()
        planner.plan(None)
        steady = Planner(SLOW_CHUNKED, chunk_streaming=True)
        steady.feed(S2 * 6)
        steady.end()
        steady.plan(None)
        assert planner.bank_needed(0.0) - steady.bank_needed(0.0) == pytest.approx(
            1.0, abs=0.05
        )
