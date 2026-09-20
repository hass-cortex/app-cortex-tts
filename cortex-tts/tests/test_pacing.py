"""The sentence buffer, the verdict and the pacer, driven without a clock.

What is pinned: a sentence is complete only once its stop arrives, a verdict
is one comparison against `STREAM_RTF`, and the pacer hands a streaming reply
one sentence at a time while a buffered reply takes everything complete.
"""

from __future__ import annotations

import pytest

from cortex_speech.pacing import (
    BUFFERED,
    STREAM_RTF,
    STREAMING,
    Pacer,
    SentenceBuffer,
    bank_needed,
    verdict,
)

S1 = "從前有一座山，山上有一間小廟。"
S2 = "廟裡住著一位老和尚和一位小和尚。"
S3 = "每天早上他們都到溪邊打水。"


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

    def test_take_all_empties_it(self) -> None:
        buffer = SentenceBuffer()
        buffer.feed(S1 + "日子")
        assert buffer.take_all() == S1 + "日子"
        assert not buffer


class TestTheVerdictIsOneComparison:
    def test_under_the_threshold_streams(self) -> None:
        assert verdict(STREAM_RTF - 0.01) == STREAMING

    def test_at_or_over_it_is_buffered(self) -> None:
        assert verdict(STREAM_RTF) == BUFFERED
        assert verdict(2.8) == BUFFERED

    def test_unmeasured_is_buffered(self) -> None:
        """Nothing measured is the one case that can never run dry."""
        assert verdict(None) == BUFFERED

    def test_the_default_is_the_recorded_hosts_line(self) -> None:
        """0.8: Hojo 40M on the HA host (0.67 bench, 0.58 production) streams,
        OmniVoice on it (5.08) does not. See ADR 0001 for the cells at 0.78
        this admits and why the wait it saves is worth them."""
        assert STREAM_RTF == 0.8
        assert verdict(0.67) == STREAMING
        assert verdict(5.08) == BUFFERED

    def test_a_host_may_move_the_line(self) -> None:
        """MOSS on the HA host measures 1.06–1.12 and, releasing audio within
        a sentence, runs dry only on replies past about a hundred seconds;
        a host that would rather not wait half a minute for the first word
        sets its threshold above that."""
        assert verdict(1.12) == BUFFERED
        assert verdict(1.12, 1.2) == STREAMING
        assert verdict(None, 1.2) == BUFFERED


class TestTheBankHoldsOnlyWhatTheRestNeeds:
    """`bank_needed`: the least lead under which no remaining sentence is
    reached by playback before it is rendered."""

    def test_one_sentence_left_needs_its_own_render_time(self) -> None:
        assert bank_needed(1.05, [4.1]) == pytest.approx(1.05 * 4.1)

    def test_a_fast_voice_gives_lead_back_sentence_by_sentence(self) -> None:
        # 0.3x: the second sentence needs 6 s of render but the first, played
        # meanwhile, hands back 0.7 × 4 s of it.
        assert bank_needed(0.3, [4.0, 20.0]) == pytest.approx(6.0 - 2.8)

    def test_a_slow_voice_loses_lead_sentence_by_sentence(self) -> None:
        # 2x: every sentence played costs another sentence's worth of lead.
        assert bank_needed(2.0, [1.0, 1.0, 1.0]) == pytest.approx(4.0)

    def test_nothing_left_needs_nothing(self) -> None:
        assert bank_needed(1.5, []) == 0.0

    def test_the_pacer_says_what_is_still_in_hand(self) -> None:
        pacer = Pacer(STREAMING)
        pacer.feed(S1 + S2 + "日子")
        assert pacer.next_request() == S1
        assert pacer.remaining() == [S2, "日子"]
        assert pacer.next_request() == S2, "looking left it in place"


class TestAStreamingReplyIsOneSentencePerRequest:
    def test_nothing_until_a_sentence_is_complete(self) -> None:
        pacer = Pacer(STREAMING)
        pacer.feed("從前有一座山，")
        assert pacer.next_request() is None
        assert not pacer.finished()

    def test_one_sentence_at_a_time_even_when_several_wait(self) -> None:
        pacer = Pacer(STREAMING)
        pacer.feed(S1 + S2 + S3)
        assert pacer.next_request() == S1
        assert pacer.next_request() == S2
        assert pacer.next_request() == S3
        assert pacer.next_request() is None

    def test_the_tail_is_a_sentence_once_the_writer_ended(self) -> None:
        pacer = Pacer(STREAMING)
        pacer.feed(S1 + "日子過得")
        assert pacer.next_request() == S1
        assert pacer.next_request() is None
        pacer.end()
        assert pacer.next_request() == "日子過得"
        assert pacer.next_request() is None
        assert pacer.finished()

    def test_text_arriving_a_character_at_a_time(self) -> None:
        pacer = Pacer(STREAMING)
        seen: list[str] = []
        for char in S1 + S2:
            pacer.feed(char)
            if (text := pacer.next_request()) is not None:
                seen.append(text)
        assert seen == [S1, S2]


class TestABufferedReplyTakesEverythingComplete:
    def test_all_complete_sentences_go_as_one_request(self) -> None:
        pacer = Pacer(BUFFERED)
        pacer.feed(S1 + S2 + "日子過得")
        assert pacer.next_request() == S1 + S2
        assert pacer.next_request() is None

    def test_the_rest_follows_once_the_writer_ended(self) -> None:
        pacer = Pacer(BUFFERED)
        pacer.feed(S1 + S2 + "日子過得")
        pacer.next_request()
        pacer.feed("很平靜。" + S3)
        pacer.end()
        assert pacer.next_request() == "日子過得很平靜。" + S3
        assert pacer.finished()

    def test_a_reply_written_in_one_go_is_one_request(self) -> None:
        pacer = Pacer(BUFFERED)
        pacer.feed(S1 + S2 + S3)
        pacer.end()
        assert pacer.next_request() == S1 + S2 + S3


class TestNothingToSay:
    @pytest.mark.parametrize("mode", [STREAMING, BUFFERED])
    def test_an_empty_reply_finishes_at_once(self, mode: str) -> None:
        pacer = Pacer(mode)
        pacer.end()
        assert pacer.next_request() is None
        assert pacer.finished()

    @pytest.mark.parametrize("mode", [STREAMING, BUFFERED])
    def test_whitespace_alone_is_nothing(self, mode: str) -> None:
        pacer = Pacer(mode)
        pacer.feed("  \n ")
        pacer.end()
        assert pacer.next_request() is None
        assert pacer.finished()
