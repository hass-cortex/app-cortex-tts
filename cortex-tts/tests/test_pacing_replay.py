"""The opening hold, checked against an independent replay of the reply.

The planner computes the hold from its own arithmetic. These tests replay the
plan it produced — rendering back to back, playing from the hold onward — and
ask what the hold *had* to be for playback never to catch the renderer. Two
things are pinned by that comparison:

- The arithmetic is exact. Across every render speed and reply length here the
  planner asks for the replay's own answer plus `MARGIN_S`, and nothing else.
  What this catches is a *shape* change — a deficit rounded up, an allowance
  that grows with the reply, a boundary paid for twice. What it cannot catch
  is anything the comparison is written in terms of, or anything the model
  here sets to zero: `MARGIN_S` is the first, and the two insurance terms —
  the fit's spread and its drift — are the second. Those are pinned in
  `test_pacing.py`, against models that carry them.
- That nothing switches strategy on its own. `planned` holds for what the
  reply needs at every render speed here, and `unheld` holds nothing at any of
  them. Nothing bounds the first of those: a hold computed never to stall,
  cut short, is a plan played in a way it was not chosen for.

The replay credits the same sentence-end allowance the planner does: a gap at
a sentence end is heard as a pause rather than as a fault, so `hold_for`
subtracts it and so must anything checking that figure.
"""

from __future__ import annotations

import pytest

from cortex_speech.pacing import PLANNED, UNHELD, Finished, Planner, RenderModel, Send
from cortex_speech.pacing.planner import MARGIN_S, SENTENCE_PAUSE_S
from cortex_speech.pacing.sentences import ends_sentence

# Ten sentences of plain Chinese, so a reply can be made longer a sentence at
# a time without changing how it reads.
SENTENCES = [
    "從前有一座山，山上有一間小廟。",
    "廟裡住著一位老和尚和一位小和尚。",
    "每天早上他們都到溪邊打水。",
    "日子過得很平靜。",
    "後來山下來了一個旅人。",
    "旅人帶著一把舊傘和一封信。",
    "老和尚請他坐下喝茶。",
    "小和尚在旁邊偷偷地看著。",
    "窗外的雨一直沒有停。",
    "他們就這樣聊到了天亮。",
]


# Chinese at 4 chars/s, as the rest of the pacing tests use, so a sentence's
# length in seconds is legible from its length in characters.
def _model(rtf: float) -> RenderModel:
    return RenderModel(
        fixed_s=0.3, per_audio=rtf, cjk_per_s=4.0, latin_per_s=14.0, samples=12
    )


def _planned(
    model: RenderModel, sentences: int, mode: str = PLANNED
) -> tuple[list[str], float]:
    """Drive a whole reply through the planner; return its batches and hold.

    `mode` is asked for and then checked: nothing here should arrive at a
    strategy the caller did not name.
    """
    planner = Planner(
        model,
        chunk_streaming=False,
        mode=mode,
        pause_s=SENTENCE_PAUSE_S,
        batch_cap_s=9.0,
    )
    planner.feed("".join(SENTENCES[:sentences]))
    planner.end()
    batches: list[str] = []
    hold: float | None = None
    decision = planner.plan(None)
    while not isinstance(decision, Finished):
        if isinstance(decision, Send):
            batches.append(decision.text)
            if hold is None:
                hold = decision.hold_wall_s
        decision = planner.plan(3.0)
    assert planner.mode == mode, f"expected {mode}, got {planner.mode}"
    assert hold is not None
    return batches, hold


def _worst_lead(model: RenderModel, batches: list[str], hold: float) -> float:
    """The least audio the listener holds, replaying this plan with this hold.

    A whole-render engine hands a batch over at its end, so the lead is lowest
    in the instant before the next one lands: the most has been played and the
    least delivered. Sentence ends carry their own pause, as they do in
    `Schedule.hold_for`.
    """
    landed: list[float] = []
    delivered_before: list[float] = []
    clock = 0.0
    delivered = 0.0
    for batch in batches:
        audio = model.audio_seconds(batch)
        clock += model.render_seconds(audio)
        landed.append(clock)
        delivered_before.append(delivered)
        delivered += audio
    started = landed[0] + hold
    worst = float("inf")
    absorbed = 0.0
    for index, at in enumerate(landed):
        if index:
            played = max(0.0, at - started)
            worst = min(worst, delivered_before[index] - played + absorbed)
        if ends_sentence(batches[index]):
            absorbed += SENTENCE_PAUSE_S
    return worst


def _hold_the_replay_needs(model: RenderModel, batches: list[str]) -> float:
    """The smallest hold whose replay never runs dry, to 0.01 s."""
    low, high = 0.0, 120.0
    assert _worst_lead(model, batches, high) >= 0.0, "120 s is not enough to hold"
    for _ in range(24):
        middle = (low + high) / 2
        if _worst_lead(model, batches, middle) >= 0.0:
            high = middle
        else:
            low = middle
    return high


# Render speeds either side of the point where the cap starts to bite, and
# reply lengths that change how many boundaries there are to pay for.
SPEEDS = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0]
LENGTHS = [4, 7, 10]


class TestTheHoldIsTheReplaysOwnAnswer:
    """What the planner asks for, against what the reply turns out to need."""

    @pytest.mark.parametrize("rtf", SPEEDS)
    @pytest.mark.parametrize("sentences", LENGTHS)
    def test_it_asks_for_the_need_and_the_margin_and_nothing_more(
        self, rtf: float, sentences: int
    ) -> None:
        """The arithmetic is exact; the only allowance is `MARGIN_S`.

        Uncapped, so this measures the computation rather than the bound. A
        planner that grew conservative — charging a spread twice, rounding a
        deficit up — widens this gap, and a reply that still sounds fine is
        not evidence that it did not.
        """
        model = _model(rtf)
        batches, asked = _planned(model, sentences)
        needed = _hold_the_replay_needs(model, batches)
        assert asked == pytest.approx(needed + MARGIN_S, abs=0.05)

    def test_the_allowance_is_half_a_second(self) -> None:
        """Pinned as a value, because the comparison above is written in it.

        Retuning it is a change to every opening wait on every host, so it
        should be a line in a diff rather than a number that drifted.
        """
        assert MARGIN_S == 0.5


class TestTheStrategyIsTheCallersAndNothingSwitchesIt:
    """Which way a written-out reply is spoken is asked for, never inferred.

    A planner that picked per reply — from a length nobody can see — would
    give the same model and the same settings two different behaviours, and
    leave anyone hearing the difference with no way to tell which they got.
    So `planned` holds for exactly as long as never running dry takes, however
    long that is, and `unheld` holds nothing. Both are the caller's word.
    """

    @pytest.mark.parametrize("rtf", [0.8, 1.6, 2.5, 3.0])
    def test_planned_holds_for_what_the_reply_needs_however_long_that_is(
        self, rtf: float
    ) -> None:
        """No bound. A hold cut short is a promise not kept, which is worse
        than a long one: the plan was grouped never to run dry."""
        model = _model(rtf)
        batches, asked = _planned(model, 10, mode=PLANNED)
        needed = _hold_the_replay_needs(model, batches)
        assert asked == pytest.approx(needed + MARGIN_S, abs=0.05)
        assert _worst_lead(model, batches, asked) >= 0.0, "it never runs dry"

    @pytest.mark.parametrize("rtf", [0.8, 1.6, 2.5, 3.0])
    def test_unheld_holds_nothing_at_any_render_speed(self, rtf: float) -> None:
        model = _model(rtf)
        _, asked = _planned(model, 10, mode=UNHELD)
        assert asked == 0.0

    @pytest.mark.parametrize("mode", [PLANNED, UNHELD])
    def test_the_whole_reply_is_spoken_whichever_was_asked_for(self, mode: str) -> None:
        """Both take the words in one go, so both must give all of them back.

        They differ in how the reply is cut and whether the opening waits —
        not in how much of it is spoken. Dispatching only one of them to the
        queued path left the other sending its first batch and stopping, with
        nine sentences of a ten-sentence reply silently dropped.
        """
        model = _model(2.5)
        batches, _ = _planned(model, 10, mode=mode)
        assert "".join(batches) == "".join(SENTENCES[:10])
        assert len(batches) > 1, "a ten-sentence reply is not one request here"

    @pytest.mark.parametrize("rtf", [0.8, 1.6, 2.5, 3.0])
    def test_a_chunk_engine_is_given_the_same_deadline(self, rtf: float) -> None:
        """Its own arithmetic, not a figure that ignores the plan.

        A chunk-streaming engine's bank is re-asked on every chunk, which is
        enough while the model keeps up. Past that it is not: at 1.6 the first
        request banks 13.0 s against the 17.0 s the rest is predicted to lose,
        so the bank alone releases nothing until a later request lands, and
        only the deadline speaks on the plan's own terms.
        """
        planner = Planner(
            _model(rtf),
            chunk_streaming=True,
            mode=PLANNED,
            pause_s=SENTENCE_PAUSE_S,
            batch_cap_s=9.0,
        )
        planner.feed("".join(SENTENCES[:10]))
        planner.end()
        decision = planner.plan(None)
        assert isinstance(decision, Send)
        assert decision.hold_bank
        assert decision.hold_wall_s > 0.0, "its own deadline, not a flat ceiling"
