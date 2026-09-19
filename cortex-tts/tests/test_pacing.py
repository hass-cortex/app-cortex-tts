"""The planner, the render model and the sentence buffer, driven without a clock.

Every figure the planner uses comes from a `RenderModel` built here to look
like a measured host: a fast whole-render engine (the 40M on a GPU), a
whole-render engine at real time with a fixed cost (OmniVoice with a clone), a
chunk-streaming engine slower than real time (MOSS on a CPU). What is pinned
is which of the three ways to speak a reply each one gets, and that the
opening request has content.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from cortex_speech import BY_ID
from cortex_speech.pacing import (
    BUFFERED,
    PLANNED,
    STREAMING,
    UNHELD,
    Finished,
    Planner,
    RenderModel,
    RenderSample,
    Send,
    SentenceBuffer,
    Wait,
    clause_pieces,
)
from cortex_speech.pacing.model import MAX_DRIFT
from cortex_speech.pacing.planner import (
    BATCH_CAP_S,
    MARGIN_S,
    _sentences_of,
)

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

# One sentence, 124 characters, no stop until the end — the shape a reply
# listing what is in a room takes. Real: it is what an assistant answered on
# the production host, and it rendered as a single request.
ENUMERATION = (
    "臥室裡目前有入口燈、大燈、小燈、螢幕掛燈、夜燈、浴室燈、窗簾、升降桌、"
    "循環扇、立扇、空氣淨化器、冷氣、木頭喇叭、掃拖機器人、冰箱、氣炸鍋、"
    "換氣扇、電磁爐、檯燈，還有門磁、人在感應、溫濕度、空氣品質和靜音、"
    "睡眠按鈕這些感應器與按鈕。"
)


def _drain(planner: Planner) -> list[Send]:
    """Every request a finished reply produces, in order."""
    sent: list[Send] = []
    while True:
        decision = planner.plan(0.0)
        if isinstance(decision, Finished):
            return sent
        assert isinstance(decision, Send), decision
        sent.append(decision)


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

    def test_a_cut_survives_the_blank_lines_between_paragraphs(self) -> None:
        """What the planner sends is sentences joined, and the joining loses
        the whitespace `_split` drops. A reply with paragraph breaks is
        ordinary, and requiring a literal prefix raised on every one of them.
        """
        buffer = SentenceBuffer()
        buffer.feed(S1 + "\n\n" + S2 + "\n\n" + S3)
        buffer.take_prefix(S1 + S2)
        assert buffer.sentences == [S3], "only the two that were sent are gone"

    def test_a_cut_that_is_not_in_the_buffer_is_still_refused(self) -> None:
        buffer = SentenceBuffer()
        buffer.feed(S1 + S2)
        with pytest.raises(ValueError):
            buffer.take_prefix(S2)

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


class TestTheRealTimeFactorNeedsALength:
    """RTF is render seconds over audio seconds, and `per_audio` is the slope.

    They agree only where the line has no fixed cost. With one, a request also
    pays `fixed_s / audio`, so the factor falls as the request grows — and a
    card that prints the slope under the heading "real-time factor" reads
    0.29 for a line whose requests actually cost 1.54 times their audio.
    """

    FLAT = RenderModel(
        fixed_s=0.0,
        per_audio=1.38,
        cjk_per_s=4.1,
        latin_per_s=14.7,
        samples=12,
        audio_ref_s=5.0,
    )
    WITH_FIXED = RenderModel(
        fixed_s=6.451,
        per_audio=0.294,
        cjk_per_s=4.1,
        latin_per_s=14.7,
        samples=11,
        audio_ref_s=5.17,
    )

    def test_no_fixed_cost_makes_the_slope_the_factor(self) -> None:
        assert self.FLAT.real_time_factor() == pytest.approx(1.38)
        assert self.FLAT.real_time_factor(60.0) == pytest.approx(1.38)

    def test_a_fixed_cost_makes_it_fall_with_the_request(self) -> None:
        model = self.WITH_FIXED
        assert model.real_time_factor(3.44) > model.real_time_factor(10.0)
        assert model.real_time_factor(60.0) > model.per_audio
        assert model.real_time_factor(10_000.0) == pytest.approx(
            model.per_audio, abs=0.01
        ), "the slope is the factor only in the limit"

    def test_it_is_quoted_at_the_length_this_host_served(self) -> None:
        model = self.WITH_FIXED
        assert model.real_time_factor() == pytest.approx(
            model.render_seconds(model.audio_ref_s) / model.audio_ref_s
        )
        assert model.real_time_factor() == pytest.approx(1.54, abs=0.01)

    def test_a_line_with_no_reference_length_falls_back_to_the_slope(
        self,
    ) -> None:
        """Rather than dividing by nothing."""
        model = replace(self.FLAT, audio_ref_s=0.0)
        assert model.real_time_factor() == model.per_audio

    def test_the_fit_records_the_length_it_was_fitted_over(self) -> None:
        samples = [
            RenderSample(
                audio_s=audio, wall_s=0.3 + 0.8 * audio, cjk=int(audio * 4.1), latin=0
            )
            for audio in (3.0, 5.0, 7.0, 9.0, 4.0, 6.0)
        ]
        fitted = RenderModel.fit(samples)
        assert fitted is not None
        assert fitted.audio_ref_s == pytest.approx(
            sum(s.audio_s for s in samples) / len(samples), abs=0.01
        )


class TestDriftIsTheLineMoving:
    """Scatter cancels over a sum; a line that is low does not.

    A hold adds up what the line says a whole reply will cost, so what breaks
    one is not how noisy the line is but whether it is systematically under.
    Measured across the eight lines one host had fitted, the within-window
    scatter was 2.0-9.1% of a render while the older half missed the newer
    half's total by up to 7.9% on lines whose scatter was 2.6%.
    """

    @staticmethod
    def _run(
        per_audio: float, count: int, *, dearer: float = 1.0
    ) -> list[RenderSample]:
        """Requests of assorted lengths, the later half dearer by `dearer`."""
        lengths = [3.0, 7.0, 5.0, 11.0, 4.0, 9.0, 6.0, 13.0, 8.0, 10.0, 5.5, 12.0]
        out = []
        for index in range(count):
            audio = lengths[index % len(lengths)]
            factor = dearer if index >= count // 2 else 1.0
            out.append(
                RenderSample(
                    audio_s=audio,
                    wall_s=(0.3 + per_audio * audio) * factor,
                    cjk=int(audio * 4.1),
                    latin=0,
                )
            )
        return out

    def test_a_line_that_holds_has_no_drift(self) -> None:
        fitted = RenderModel.fit(self._run(0.8, 12))
        assert fitted is not None
        assert fitted.drift == 0.0

    def test_a_line_the_reply_outruns_carries_the_shortfall(self) -> None:
        fitted = RenderModel.fit(self._run(0.8, 12, dearer=1.10))
        assert fitted is not None
        assert 0.05 < fitted.drift < 0.15, fitted.drift

    def test_a_line_that_over_predicts_needs_no_insurance(self) -> None:
        """Only under-prediction costs the listener anything."""
        fitted = RenderModel.fit(self._run(0.8, 12, dearer=0.85))
        assert fitted is not None
        assert fitted.drift == 0.0

    def test_too_few_samples_say_nothing(self) -> None:
        fitted = RenderModel.fit(self._run(0.8, 6, dearer=1.10))
        assert fitted is not None
        assert fitted.drift == 0.0, "no older half to fit, no newer half to test"

    def test_one_bad_window_cannot_buy_a_minute_of_silence(self) -> None:
        fitted = RenderModel.fit(self._run(0.8, 12, dearer=3.0))
        assert fitted is not None
        assert fitted.drift == MAX_DRIFT

    @pytest.mark.parametrize("chunked", [False, True])
    def test_the_bank_does_not_take_the_insurance(self, chunked: bool) -> None:
        """Insurance is the deadline's; the bank has better evidence.

        The deadline is settled once, before a byte exists, so it is costed on
        the line made dearer by its drift. The bank is re-asked as the reply
        runs and scales the line by what this reply has actually cost — asking
        it to carry the drift as well charges the same doubt twice, and the
        reply waits for a shortfall it can already see is not happening.
        """
        reply = (S1 + S2 + S3 + S4) * 3
        asked = {}
        for drift in (0.0, 0.2):
            model = replace(
                SLOW_CHUNKED if chunked else REALTIME_WHOLE,
                drift=drift,
                spread_s=0.2,
                spread_ref_s=10.0,
            )
            planner = Planner(model, chunk_streaming=chunked, mode=PLANNED)
            planner.feed(reply)
            planner.end()
            decision = planner.plan(None)
            assert isinstance(decision, Send)
            asked[drift] = (decision.hold_wall_s, planner.bank_needed(0.0, 0.0))

        assert asked[0.2][0] > asked[0.0][0], "the deadline carries it"
        assert asked[0.2][1] == pytest.approx(asked[0.0][1]), "the bank does not"

    def test_the_hold_is_widened_in_proportion_to_it(self) -> None:
        """And the widening is of the whole plan, which is what drift is of."""
        reply = (S1 + S2 + S3 + S4) * 3
        holds = {}
        for drift in (0.0, 0.05):
            model = replace(SLOW_CHUNKED, drift=drift)
            planner = Planner(model, chunk_streaming=True, mode=PLANNED)
            planner.feed(reply)
            planner.end()
            decision = planner.plan(None)
            assert isinstance(decision, Send)
            holds[drift] = decision.hold_wall_s
        assert holds[0.05] > holds[0.0]
        total = sum(
            SLOW_CHUNKED.render_seconds(SLOW_CHUNKED.audio_seconds(text))
            for text in (S1, S2, S3, S4) * 3
        )
        assert holds[0.05] - holds[0.0] == pytest.approx(0.05 * total, rel=0.25)


class TestTheSpreadIsChargedByTheRenderAtRisk:
    """One figure over a sample of requests, charged per request by size.

    Measured on the standalone host's own stored samples, the residual is
    closer to proportional than flat: across four cost lines it grew with the
    render (MOSS r=+0.67, the OmniVoice clone r=+0.38), and on the line where
    the wait hurt most the relative error was flat instead — 21.9% of the
    render over the short half of its samples against 20.7% over the long.
    Charged flat, a 2 s render paid what a 10 s one was worth: on one
    two-request reply that was 1.65 s of a 5.07 s wait, for 7.6 s of speech.
    """

    LINE = RenderModel(
        fixed_s=1.0,
        per_audio=0.6,
        cjk_per_s=5.0,
        latin_per_s=14.0,
        samples=20,
        spread_s=1.2,
        spread_ref_s=4.0,
    )

    def test_a_request_of_the_sample_average_pays_what_it_did(self) -> None:
        assert self.LINE.spread_for(4.0) == pytest.approx(1.2)

    def test_a_shorter_render_pays_less_and_a_longer_one_more(self) -> None:
        assert self.LINE.spread_for(2.0) == pytest.approx(0.6)
        assert self.LINE.spread_for(8.0) == pytest.approx(2.4)

    def test_it_stops_doubling_because_a_hold_has_to_end(self) -> None:
        """Past the lengths measured, proportion is an extrapolation."""
        assert self.LINE.spread_for(100.0) == pytest.approx(2.4)

    def test_a_line_that_never_said_pays_the_flat_figure(self) -> None:
        """`spread_ref_s` is 0 on a borrowed or one-point line."""
        flat = replace(self.LINE, spread_ref_s=0.0)
        assert flat.spread_for(0.5) == pytest.approx(1.2)
        assert flat.spread_for(50.0) == pytest.approx(1.2)

    def test_the_fit_records_what_it_measured_the_spread_over(self) -> None:
        model = RenderModel.fit(
            [
                RenderSample(audio_s=a, wall_s=1.0 + 0.6 * a, cjk=int(a * 5), latin=0)
                for a in (2.0, 4.0, 6.0, 8.0)
            ]
        )
        assert model is not None
        # The mean of 2.2, 3.4, 4.6 and 5.8.
        assert model.spread_ref_s == pytest.approx(4.0, abs=0.05)

    def test_the_hold_is_what_pays_it(self) -> None:
        """A plan risking a short render is held less than one risking a long.

        `Schedule.hold_for` is where the allowance lands. `speaks_at` deliberately
        leaves it out: it compares plans, and an allowance that shrinks with
        the plan is not the listener hearing anything sooner.
        """
        planner = Planner(self.LINE, chunk_streaming=False)
        flat = Planner(
            replace(self.LINE, spread_ref_s=0.0),
            chunk_streaming=False,
        )
        for one in (planner, flat):
            one.feed(S1 + S2)
            one.end()
            assert isinstance(one.plan(None), Send)
        batches = [planner._rendering, *planner._queued]
        assert planner._when.of(planner._model, batches)[2] < self.LINE.spread_ref_s
        assert planner._when.hold_for(planner._model, batches) < flat._when.hold_for(
            flat._model, [flat._rendering, *flat._queued]
        )

    def test_a_finer_cut_does_not_look_sooner_for_carrying_less_of_it(self) -> None:
        """A chunk-streaming engine is audible at its fixed cost, any cut.

        Charging the spread by the render at risk made a finer plan score
        better on `speaks_at` purely because its allowance was smaller —
        measured on MOSS, thirteen requests chosen over one, for 0.09 s that
        nobody could hear.
        """
        chunked = RenderModel(
            fixed_s=0.23,
            per_audio=0.31,
            cjk_per_s=4.9,
            latin_per_s=14.0,
            samples=12,
            spread_s=0.07,
            spread_ref_s=2.0,
        )
        planner = Planner(chunked, chunk_streaming=True)
        sentences = _sentences_of(S1 + S2 + S3 + S4)
        whole = planner._when.group(planner._model, sentences, 1e9)
        split = planner._when.group(planner._model, sentences, 2.0)
        assert len(whole) < len(split)
        assert planner._when.speaks_at(planner._model, whole) == pytest.approx(
            planner._when.speaks_at(planner._model, split), abs=1e-6
        )


class TestBufferedIsAskedForNeverConcluded:
    def test_asking_for_it_gets_one_request_however_long(self) -> None:
        """The escape hatch does nothing, and that is the whole of its value.

        Cutting would shorten the total render — measured on OmniVoice, 20.9 s
        of speech cost 27.9 s whole against 16.5 s in thirds — but a held
        reply has no earlier first word to buy with it, and the price is an
        unmeasured change in how the reply sounds.
        """
        whole = (S1 + S2 + S3 + S4) * 3
        planner = Planner(FAST_WHOLE, chunk_streaming=False, mode=BUFFERED)
        planner.feed(whole)
        assert planner.mode == BUFFERED
        assert isinstance(planner.plan(None), Wait)
        planner.end()
        batches = _drain(planner)
        assert [b.text for b in batches] == [whole]
        assert batches[0].hold_all

    def test_an_unmeasured_model_is_paced_instead(self) -> None:
        """Its own first request is this host, in this voice, a moment ago."""
        planner = Planner(None, chunk_streaming=False)
        planner.feed(ENUMERATION)
        assert planner.mode != BUFFERED
        assert isinstance(planner.plan(None), Wait), "nothing to plan until it ends"
        planner.end()
        batches = _drain(planner)
        assert planner.mode == PLANNED
        assert len(batches) > 1
        assert "".join(b.text for b in batches) == ENUMERATION
        assert batches[0].hold_bank and not batches[0].hold_all

    def test_it_releases_nothing_until_that_request_lands(self) -> None:
        planner = Planner(None, chunk_streaming=False)
        planner.feed(ENUMERATION)
        planner.end()
        first = planner.plan(None)
        assert isinstance(first, Send)
        assert planner.bank_needed(0.0) == float("inf")
        planner.rendered(audio_s=8.0, wall_s=5.0)
        assert planner.bank_needed(0.0) < 8.0, "the line it just learned"

    def test_what_it_learns_is_this_voice_not_the_prior(self) -> None:
        planner = Planner(None, chunk_streaming=False)
        planner.feed(S1)
        planner.end()
        sent = planner.plan(None)
        assert isinstance(sent, Send)
        planner.rendered(audio_s=3.0, wall_s=2.1)  # 15 CJK in 3 s
        assert planner._model is not None
        assert planner._model.per_audio == pytest.approx(0.7)
        assert planner._model.cjk_per_s == pytest.approx(5.0, abs=0.1)


class TestTheOpeningIsHeldForWhatThePlanNeeds:
    """The opening waits for the render, and nothing shortens it.

    A ceiling over the opening decides nothing worth having. The opening a
    gaining model asks for is the margin and the spread, 0.81-0.89 s across
    every line these two hosts have fitted, so one loose enough to let those
    through never bites and one tight enough to bite refuses every model. What
    it does reach is the far side: a hold computed never to stall, cut short —
    which is the one thing holding at all is for. A reply too slow to wait out is `unheld`, chosen by
    the person who can hear it.
    """

    def test_a_model_that_cannot_gain_lead_is_planned_and_holds_what_it_needs(
        self,
    ) -> None:
        """The real-time factor decides admission; the plan decides the wait."""
        slow = RenderModel(
            fixed_s=2.0, per_audio=1.6, cjk_per_s=4.0, latin_per_s=14.0, samples=12
        )
        planner = Planner(slow, chunk_streaming=False)
        planner.feed((S1 + S2 + S3 + S4) * 4)
        planner.end()
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert planner.mode == PLANNED
        assert decision.hold_wall_s > 4.0, "the hold is what it needs to be"

    def test_a_short_hold_is_passed_through(self) -> None:
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        planner.feed(S1 + S2 + S3)
        planner.end()
        assert isinstance(planner.plan(None), Send)
        held = planner._when.hold_for(
            planner._model, [planner._rendering, *planner._queued]
        )
        assert 0.0 < held < 4.0

    def test_a_paced_reply_carries_a_deadline_and_not_only_a_bank(self) -> None:
        """The bank is only re-asked on a boundary; the deadline is between them.

        A whole-render engine hands over one batch at a time, so a release
        waiting on the bank can only happen where a request lands. Measured on
        OmniVoice against its own fitted line, that put the first word at
        8.73 s where the same plan's arithmetic allowed 5.62 s.
        """
        planner = Planner(REALTIME_WHOLE, chunk_streaming=False)
        planner.feed(S1 + S2 + S3 + S4)
        planner.end()
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.hold_bank, "still released early if the render runs ahead"
        assert decision.hold_wall_s > 0.0
        assert decision.hold_wall_s == pytest.approx(
            planner._when.hold_for(
                planner._model, [planner._rendering, *planner._queued]
            ),
            abs=1e-3,
        )

    def test_a_chunk_streaming_reply_carries_one_too(self) -> None:
        """Its own arithmetic, like a whole-render reply's.

        Its bank is re-asked on every chunk, which is enough while the model
        keeps up — measured across factors 0.4 to 1.2 the two agree to within
        the sampling. This model does not keep up, and there the bank alone
        releases nothing until a later request lands: the deadline is what
        speaks on the plan's own terms.
        """
        planner = Planner(SLOW_CHUNKED, chunk_streaming=True)
        planner.feed(S1 + S2 + S3 + S4)
        planner.end()
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert planner.mode == PLANNED, "slower than real time, so it cannot stream"
        assert decision.hold_bank, "still released early if the render runs ahead"
        assert decision.hold_wall_s == pytest.approx(
            planner._when.hold_for(
                planner._model, [planner._rendering, *planner._queued]
            ),
            abs=1e-3,
        )

    def test_an_unmeasured_reply_has_no_deadline_to_carry(self) -> None:
        """There is no line yet, so only the bank can end the wait.

        `bank_needed` is infinite until the reply's own first request lands
        and says what a request that size costs here; from then it is finite
        and the next check releases.
        """
        planner = Planner(None, chunk_streaming=False)
        planner.feed(S1 + S2 + S3 + S4)
        planner.end()
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.hold_bank
        assert decision.hold_wall_s == 0.0

    def test_a_model_at_real_time_cannot_stream_an_unwritten_reply(self) -> None:
        """Admission is the real-time factor and nothing else.

        A model that does not gain lead per request would have to bank against
        a reply whose length nobody yet knows, so it is planned once written.
        """
        planner = Planner(REALTIME_WHOLE, chunk_streaming=False)
        planner.feed(S1 + S2 + S3 + S4)
        while isinstance(planner.plan(None), Wait):
            planner.feed(S1)
            if planner.mode != STREAMING:
                break
        assert planner.mode == PLANNED, "it gains no lead, so it cannot stream"


class TestASentenceEndCarriesItsOwnPause:
    """Playback catching the renderer at a sentence end is a pause, not a gap.

    Measured on one 313-character reply, OmniVoice with a clone: holding until
    nothing could ever run dry put the first word at 6.29 s; letting each
    sentence end carry its own silence put it at 4.87 s, with one gap of
    0.55 s, after the first sentence.
    """

    @staticmethod
    def _hold(pause: float, model: RenderModel = REALTIME_WHOLE) -> float:
        planner = Planner(model, chunk_streaming=False, pause_s=pause)
        planner.feed(S1 + S2 + S3 + S4)
        planner.end()
        assert isinstance(planner.plan(None), Send)
        return planner._when.hold_for(
            planner._model, [planner._rendering, *planner._queued]
        )

    def test_the_allowance_shortens_the_opening(self) -> None:
        assert self._hold(1.5) < self._hold(0.0)

    def test_none_of_it_is_spent_before_the_first_boundary(self) -> None:
        """There is no boundary yet, so the first request is covered in full."""
        planner = Planner(REALTIME_WHOLE, chunk_streaming=False, pause_s=99.0)
        planner.feed(S1)
        planner.end()
        assert isinstance(planner.plan(None), Send)
        needed, first, _ = planner._when.of(planner._model, [planner._rendering or ""])
        assert needed == pytest.approx(first)

    def test_a_clause_cut_earns_nothing(self) -> None:
        """Silence inside a sentence is a fault, so it cannot be budgeted for."""
        generous = Planner(REALTIME_WHOLE, chunk_streaming=False, pause_s=4.0)
        pieces = clause_pieces("一二三四五六七八，九十一二三四五六，七八九十一二三四")
        assert len(pieces) == 3 and not any(
            p.rstrip()[-1] in "。！？" for p in pieces[:-1]
        )
        strict = Planner(REALTIME_WHOLE, chunk_streaming=False, pause_s=0.0)
        assert generous._when.of(generous._model, pieces)[0] == pytest.approx(
            strict._when.of(strict._model, pieces)[0]
        )

    def test_a_chunked_engine_earns_nothing_either(self) -> None:
        """Its audio arrives continuously, so a dry moment falls anywhere."""
        batches = [S1, S2, S3]
        generous = Planner(SLOW_CHUNKED, chunk_streaming=True, pause_s=4.0)
        strict = Planner(SLOW_CHUNKED, chunk_streaming=True, pause_s=0.0)
        assert generous._when.of(generous._model, batches)[0] == pytest.approx(
            strict._when.of(strict._model, batches)[0]
        )


class TestTheCapBoundsAStreamedBatchToo:
    """A lead is not a licence to hand the model the rest of the reply.

    `_streaming_next` groups the pieces a lead can cover, and `cut_to_cap`
    bounds each piece rather than the group — so a fast model with a wide lead
    reaches the end of the reply in one request. The cap is where the fitted
    line stops describing the model, and past it the audio itself is not
    reliable: measured on MOSS, which declares fifteen seconds, a 269-character
    batch came back as 163 characters' worth with the middle missing.
    """

    CAP = 15.0
    FAST_CHUNKED = RenderModel(
        fixed_s=0.3,
        per_audio=0.45,
        cjk_per_s=5.45,
        latin_per_s=14.0,
        samples=12,
        spread_s=0.156,
        spread_ref_s=5.0,
    )
    REPLY = (S1 + S2 + S3 + S4) * 4

    def _batches(self, mode: str) -> list[str]:
        planner = Planner(
            self.FAST_CHUNKED,
            chunk_streaming=True,
            mode=mode,
            batch_cap_s=self.CAP,
        )
        planner.feed(self.REPLY)
        planner.end()
        sent: list[str] = []
        # A lead far wider than the reply, which is what lets one request grow.
        decision = planner.plan(60.0)
        while isinstance(decision, Send):
            sent.append(decision.text)
            decision = planner.plan(60.0)
        return sent

    def test_no_streamed_request_carries_more_than_the_cap(self) -> None:
        batches = self._batches(STREAMING)
        worst = max(self.FAST_CHUNKED.audio_seconds(text) for text in batches)
        assert worst <= self.CAP
        assert len(batches) > 1, "a lead this wide would otherwise take it whole"

    def test_planned_was_already_bounded_the_same_way(self) -> None:
        worst = max(
            self.FAST_CHUNKED.audio_seconds(text) for text in self._batches(PLANNED)
        )
        assert worst <= self.CAP

    @pytest.mark.parametrize("mode", [STREAMING, PLANNED])
    def test_the_reply_survives_being_cut(self, mode: str) -> None:
        assert "".join(self._batches(mode)) == self.REPLY


class TestTheBatchCapBelongsToTheModel:
    """Where a fitted line stops describing a model is the model's own.

    Nine seconds is OmniVoice's: measured against a line fitted from requests
    under 8 s it was 19% over by 10.7 s. MOSS-TTS-Nano stayed within 6% out to
    28 s, and holding it to nine cost what the cap's own reasoning had not
    counted — not render time, which is 0.05 s a boundary there, but cuts: a
    sentence over the cap is split at its clause marks and every segment is
    terminated, so those are spoken as full stops.
    """

    CHUNKED = RenderModel(
        fixed_s=0.23,
        per_audio=0.31,
        cjk_per_s=4.9,
        latin_per_s=14.0,
        samples=12,
        spread_s=0.07,
        spread_ref_s=2.0,
    )

    @staticmethod
    def _plan(cap: float, model: RenderModel, chunked: bool) -> list[str]:
        planner = Planner(model, chunk_streaming=chunked, batch_cap_s=cap)
        planner.feed((S1 + S2 + S3 + S4) * 3)
        planner.end()
        assert isinstance(planner.plan(None), Send)
        return [planner._rendering or "", *planner._queued]

    def test_a_model_that_breaks_later_is_cut_less(self) -> None:
        near = self._plan(9.0, self.CHUNKED, True)
        far = self._plan(15.0, self.CHUNKED, True)
        assert len(far) < len(near)

    def test_the_catalog_carries_it_and_the_default_is_the_earliest_break(
        self,
    ) -> None:
        assert BY_ID["omnivoice"].batch_cap_s == BATCH_CAP_S
        assert BY_ID["moss-nano"].batch_cap_s == 15.0

    def test_a_model_held_to_the_default_is_unchanged(self) -> None:
        """OmniVoice is where the nine came from; nothing about it moves."""
        assert self._plan(BATCH_CAP_S, FAST_WHOLE, False) == self._plan(
            BATCH_CAP_S, FAST_WHOLE, False
        )
        for batch in self._plan(BATCH_CAP_S, FAST_WHOLE, False):
            assert FAST_WHOLE.audio_seconds(batch) <= BATCH_CAP_S + 1e-6


class TestASentenceLongerThanOneRequest:
    """Grouping can join sentences but never cut below one.

    A reply that lists what is in a room is written as one sentence with
    clause marks and a single stop, so every plan handed it to the renderer
    whole — the case the cap exists to prevent, arriving by the one route that
    bypassed it. Its own clauses are the pieces; grouping joins them back up
    to the cap like any others.
    """

    @pytest.mark.parametrize("model", [REALTIME_WHOLE, FAST_WHOLE])
    def test_a_paced_reply_is_cut_at_clause_marks(self, model: RenderModel) -> None:
        planner = Planner(model, chunk_streaming=False)
        planner.feed(ENUMERATION)
        planner.end()
        batches = _drain(planner)
        assert len(batches) > 1
        assert "".join(b.text for b in batches) == ENUMERATION
        assert all(model.audio_seconds(b.text) <= BATCH_CAP_S for b in batches)

    @pytest.mark.parametrize("lead", [3.0, 0.6])
    def test_streaming_is_capped_too(self, lead: float) -> None:
        """The lead bounds a batch only while there is a boundary to stop at.

        A reply written as one long sentence has none until its stop arrives,
        so what is left of it reaches `_streaming_next` as a single piece
        however small the lead — 15.9 s of it, before the cap applied here.
        """
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        sent, current = [], None
        for i in range(0, len(ENUMERATION), 6):
            planner.feed(ENUMERATION[i : i + 6])
            while isinstance(decision := planner.plan(current), Send):
                sent.append(decision.text)
                current = lead
        planner.end()
        sent.extend(d.text for d in _drain(planner))
        assert planner.mode == STREAMING
        assert "".join(sent) == ENUMERATION
        assert all(FAST_WHOLE.audio_seconds(t) <= BATCH_CAP_S for t in sent)

    def test_a_sentence_with_no_clause_mark_is_sent_whole(self) -> None:
        """There is nowhere to cut, and a cut mid-word is worse than the cost."""
        run_on = "啊" * 200
        planner = Planner(FAST_WHOLE, chunk_streaming=False)
        planner.feed(run_on + "。")
        planner.end()
        batches = _drain(planner)
        assert [b.text for b in batches] == [run_on + "。"]


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
        assert planner.mode == PLANNED
        assert isinstance(planner.plan(0.0), Finished)


class TestInsistingOnStreamingDoesNotDependOnTiming:
    """The same reply must reach the same delivery however it was handed over.

    The transport creates its reader task and then plans before that task has
    run, so whether `end` had arrived by the first decision was a matter of
    which coroutine the loop picked up first. Measured against the fake engine
    before this: ten replies on a fresh app planned every time and eight on a
    warm one streamed seven, from identical input. A delivery nobody can
    reproduce is one nobody can compare, which is the whole of what insisting
    on one is for.
    """

    REPLY = S1 + S2 + S3 + S4

    def _drive(self, *, end_first: bool, mode: str = STREAMING) -> tuple[str, str]:
        """Return the mode arrived at and every byte of text sent."""
        planner = Planner(FAST_WHOLE, chunk_streaming=False, mode=mode)
        sent: list[str] = []

        def drain() -> None:
            decision = planner.plan(3.0)
            while isinstance(decision, Send):
                sent.append(decision.text)
                decision = planner.plan(3.0)

        for sentence in (S1, S2, S3, S4):
            planner.feed(sentence)
            if not end_first:
                drain()
        planner.end()
        drain()
        return planner.mode, "".join(sent)

    def test_end_before_the_first_decision_still_streams(self) -> None:
        mode, text = self._drive(end_first=True)
        assert mode == STREAMING
        assert text == self.REPLY, "and not one character of it was dropped"

    def test_end_after_the_first_decision_streams_the_same_way(self) -> None:
        assert self._drive(end_first=False)[0] == STREAMING

    @pytest.mark.parametrize("mode", [PLANNED, UNHELD, BUFFERED])
    def test_the_other_insisted_modes_were_already_stable(self, mode: str) -> None:
        early, late = (
            self._drive(end_first=True, mode=mode),
            self._drive(end_first=False, mode=mode),
        )
        assert early[0] == late[0] == mode
        assert early[1] == late[1] == self.REPLY

    def test_a_reply_too_short_to_open_is_planned_rather_than_waited_for(self) -> None:
        """Insisting is honoured as far as the reply allows, and no further.

        Below the opening floor there is no first batch to send, and the words
        are all in — so there is nothing left to wait for. Returning `Wait`
        here would hang the reply on a buffer that will never grow.
        """
        planner = Planner(FAST_WHOLE, chunk_streaming=False, mode=STREAMING)
        planner.feed("好的。")
        planner.end()
        assert isinstance(planner.plan(None), Send)
        assert planner.mode == PLANNED


class TestStreaming:
    def test_the_next_batch_is_the_largest_the_lead_covers(self) -> None:
        # A cap out of the way, because what is under test is the lead. The
        # two bounds are separate and `TestTheCapBoundsAStreamedBatchToo`
        # covers the other one; at the 9 s default this example would be the
        # cap's answer rather than the lead's, S2+S3+S4 being 9.25 s.
        planner = Planner(FAST_WHOLE, chunk_streaming=False, batch_cap_s=60.0)
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
    """OmniVoice with a clone, MOSS on a CPU: planned, never streamed."""

    @pytest.mark.parametrize(
        ("model", "chunked"), [(REALTIME_WHOLE, False), (SLOW_CHUNKED, True)]
    )
    def test_waits_for_the_whole_reply(self, model: RenderModel, chunked: bool) -> None:
        planner = Planner(model, chunk_streaming=chunked)
        planner.feed(S1 + S2)
        assert isinstance(planner.plan(None), Wait)
        assert planner.mode == PLANNED
        planner.feed(S3 + S4)
        assert isinstance(planner.plan(None), Wait)
        planner.end()
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.text.startswith(S1)
        assert not decision.hold_all

    @pytest.mark.parametrize(
        ("model", "chunked"),
        [
            (REALTIME_WHOLE, False),
            (SLOW_CHUNKED, True),
            (
                RenderModel(
                    fixed_s=0.0,
                    per_audio=0.8,
                    cjk_per_s=4.0,
                    latin_per_s=14.0,
                    samples=12,
                ),
                False,
            ),
        ],
    )
    def test_no_other_grouping_speaks_sooner(
        self, model: RenderModel, chunked: bool
    ) -> None:
        """The plan is chosen by asking, so nothing else may beat it.

        This is the specification the constants used to approximate: soonest
        first word, with the lead never falling below the margin. It holds for
        an engine that gains lead, one that does not, and one that is audible
        mid-request, without any of them being named here.
        """
        planner = Planner(model, chunk_streaming=chunked)
        planner.feed(S2 * 6)
        planner.end()
        first = planner.plan(None)
        assert isinstance(first, Send)
        chosen = [first.text] + list(planner._queued)
        assert "".join(chosen) == S2 * 6, "nothing is lost between requests"

        # A held reply is judged on when it is audible after its hold; an
        # unheld one on when its first batch lands. Each is optimal for the
        # question it was grouped to answer, and neither is for the other's.
        held = planner.mode != UNHELD
        assert first.hold_bank is held
        audible = planner._when.speaks_at if held else planner._when.first_audio_at
        sentences = [S2] * 6
        ours = audible(planner._model, chosen)
        for limit in (2.0, 4.0, 6.0, 8.0, BATCH_CAP_S):
            rival = planner._when.group(planner._model, sentences, limit)
            assert ours <= audible(planner._model, rival) + 1e-6, limit

    def test_a_tie_goes_to_the_fewest_boundaries(self) -> None:
        """Every cut speaks at the same moment on a chunk-streaming engine.

        It is audible from its fixed cost whatever the grouping, so the search
        finds a tie and the tie has to be broken on what else a boundary
        costs. Measured on MOSS, the same story in one-sentence requests fell
        41% behind playback against 4.1% in nine-second ones.
        """
        fast = RenderModel(
            fixed_s=0.0, per_audio=0.36, cjk_per_s=4.0, latin_per_s=14.0, samples=12
        )
        planner = Planner(fast, chunk_streaming=True)
        planner.feed(S2 * 6)
        planner.end()
        first = planner.plan(None)
        assert isinstance(first, Send)
        chosen = [first.text] + list(planner._queued)
        assert planner._when.speaks_at(planner._model, chosen) == pytest.approx(
            planner._when.speaks_at(
                planner._model, planner._when.group(planner._model, [S2] * 6, 4.0)
            ),
            abs=0.01,
        ), "the cuts really do tie"
        assert len(chosen) == len(
            planner._when.group(planner._model, [S2] * 6, BATCH_CAP_S)
        ), "so the one with fewer boundaries is taken — the cap is the fewest"

    def test_the_search_stops_where_the_line_stops_being_true(self) -> None:
        """One request past the valley reads as faster, and is not.

        The fit is a line, and a request long enough to leave the valley costs
        more than a line can express — measured on MOSS, the same 55-second
        story fell 4.1% behind playback in nine-second requests and 17.8% in
        one. Searching past the cap would be optimising against a model that
        is knowingly wrong there, so the search does not look.
        """
        planner = Planner(SLOW_CHUNKED, chunk_streaming=True)
        planner.feed(S2 * 6)
        planner.end()
        first = planner.plan(None)
        assert isinstance(first, Send)
        chosen = [first.text] + list(planner._queued)
        whole = planner._when.group(planner._model, [S2] * 6, 24.0)
        assert len(whole) == 1
        assert planner._when.speaks_at(planner._model, whole) < planner._when.speaks_at(
            planner._model, chosen
        )
        assert max(SLOW_CHUNKED.audio_seconds(b) for b in chosen) <= BATCH_CAP_S

    def test_a_second_batch_that_lands_in_time_needs_only_the_margin(self) -> None:
        planner = Planner(REALTIME_WHOLE, chunk_streaming=False)
        planner.feed(S1 + S2 + S3 + S2)
        planner.end()
        first = planner.plan(None)
        assert isinstance(first, Send)
        assert first.hold_bank
        chosen = [first.text] + list(planner._queued)
        # Whatever the cut, a request that lands before the one before it has
        # played out costs the listener only the margin and the spread.
        assert planner._when.hold_for(planner._model, chosen) >= MARGIN_S

    def test_a_dear_opening_still_streams_when_the_lead_is_gained(self) -> None:
        """A dear opening is not a reason to refuse a model that keeps up.

        Admission is the real-time factor alone. Weighing the opening's own
        length as well reaches nothing any line here fits: a gaining model
        opens on the margin and the spread, 0.81-0.89 s across every line
        these two hosts have fitted, and it takes a spread of 1.652 s to push
        one past four seconds — the figure `docs/models.md` records as what
        back-to-back benchmark requests do to a fit, against 0.198 s once the
        window refilled with ordinary replies. This model gains lead on every
        request, so it never stalls; the wait is what the render costs, and
        waiting for a render is not a failure.
        """
        heavy = RenderModel(
            fixed_s=3.0,
            per_audio=0.2,
            cjk_per_s=4.0,
            latin_per_s=14.0,
            samples=9,
            spread_s=1.652,
            spread_ref_s=5.0,
        )
        planner = Planner(heavy, chunk_streaming=True)
        for _ in range(20):
            planner.feed(S1)
            if not isinstance(planner.plan(None), Wait):
                break
        assert planner.mode == STREAMING


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
        assert planner.mode == PLANNED
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
    """A planned reply releases when the bank covers what is still to be lost."""

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
        batch = SLOW_CHUNKED.audio_seconds(S2 * 2)
        planner.rendered(batch, SLOW_CHUNKED.render_seconds(batch) * 2)
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
        # The request already rendering has paid its fixed part; every one
        # behind it still carries one. How many there are is the cap's to say.
        spoken = [dear.audio_seconds(b) for b in [planner._rendering, *planner._queued]]
        in_flight = max(0.0, dear.per_audio - 1.0) * spoken[0]
        queued = sum(dear.deficit(a, chunk_streaming=True) for a in spoken[1:])
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
        planner.feed(S2 * 4)  # the cap groups these into two batches
        planner.end()
        first = planner.plan(None)
        assert isinstance(first, Send)
        assert first.hold_bank
        batch = SLOW_CHUNKED.audio_seconds(first.text)  # 8 s of speech
        mine = 0.15 * batch  # its fixed part is already paid
        theirs = 0.3 + 0.15 * batch  # the next one has not started
        # Nothing produced yet: this batch loses its share, the next all of its.
        assert planner.bank_needed(0.0) == pytest.approx(mine + theirs + 0.5, abs=0.05)
        # Half of this batch produced: its remaining loss halves.
        assert planner.bank_needed(batch / 2) == pytest.approx(
            mine / 2 + theirs + 0.5, abs=0.05
        )
        # On to the last batch, nothing produced: only its own loss.
        assert isinstance(planner.plan(0.0), Send)
        assert planner.bank_needed(0.0) == pytest.approx(mine + 0.5, abs=0.05)
        assert isinstance(planner.plan(0.0), Finished)
        assert planner.bank_needed(0.0) == 0.0

    def test_a_whole_engine_counts_the_deepest_point(self) -> None:
        planner = Planner(REALTIME_WHOLE, chunk_streaming=False)
        planner.feed(S2 * 6)
        planner.end()
        first = planner.plan(None)
        assert isinstance(first, Send)
        # The deepest point of whatever plan was chosen, which is what a whole
        # render engine must have banked: every batch lands after the ones
        # before it have played, so the bank covers the largest of those gaps.
        chosen = [first.text] + list(planner._queued)
        deepest, _, _ = planner._when.of(planner._model, chosen)
        assert planner.bank_needed(0.0) == pytest.approx(
            deepest + MARGIN_S + REALTIME_WHOLE.spread_s, abs=0.05
        )

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
