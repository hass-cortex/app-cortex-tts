# ADR 0002 — The text pipeline is keyed by language, and a pass that can misjudge is opt-in

Status: accepted.

## Context

The models ship no text front-end. Left alone they read `26.5°C` as noise,
`110` as a quantity, Traditional glyphs with the wrong readings and Taiwan
words with mainland readings — 32% character error rate against 4% with the
pipeline, measured on the same replies. The pipeline is therefore the product,
and every rule in it is a place to be wrong in a way that produces confident
audio rather than an error.

## Decision

- **The locale is chosen by the request's `language` tag**, and sniffing the
  script is only the fallback for a request that carries none. A language
  with no locale gets `text/generic.py`: numbers through `num2words`, CLDR
  units and dates through `babel`, a sentence-final stop, and nothing else —
  never another language's words.
- **A rewrite only one language needs lives with that locale** (script
  conversion and Taiwan readings exist only under `zh`) and is absent, not
  switched off, everywhere else. `/api/preview` lists only the switches the
  language has.
- **Pass order is a contract owned by `text/passes.py`.** Normalisation emits
  Traditional number words, so it precedes script conversion; Taiwan
  readings and their boundary-word list are keyed by the Simplified form, so
  they run last. Within a normaliser a construct (a unit, a clock colon, a
  date) claims its number before the bare-number pass can read the digit as
  a quantity.
- **A pass that can misjudge is off unless the caller says otherwise.**
  Bare-number expansion is off in every locale: `撥打 110`, `302號房`,
  `RTX 4090` and `John 3:16` were all read as quantities, and wrong misleads
  where unread merely goes unheard. Script conversion is glyph-for-glyph and
  a number with a unit says what it is, so those run by default.
- **Taiwan readings run by default because their one way to misjudge is
  decided per occurrence.** The table is keyed by words and the text is not
  segmented, so an entry can match a span that is not a word there: `在为`
  matched across `正在 | 为你` and the model read wéi where the sentence says
  wèi, on the commonest shape an Assist reply has. `readings._straddles`
  offers each edge character to its neighbour and the more common word keeps
  it. Dropping entries could not substitute: 5,200 of the 5,214 can straddle
  in principle, and the ones that do are not rare — `在为` occurs 100 times in
  the corpus against `夕阳`'s 116.
- **Every segment ends in sentence-final punctuation.** Without it the 40M
  invented a syllable, measured.
- **What depends on the character is keyed by script, not by language.**
  Where a sentence ends and how fast a script is spoken are properties of
  the characters — 「।」 closes a sentence whoever wrote it, one 漢字 is one
  syllable in Japanese as in Chinese — so `text/scripts.py` declares them
  once and no request has to name its language for them to apply: mixed
  text, an untagged request and a reply arriving in pieces read the same
  table. The terminators are the UAX #29 `STerm` members of the scripts a
  catalog model can reach; the batch splitter, the live sentence buffer and
  the terminator pass derive from that one set and cannot disagree. The
  speech-rate priors are one per script, and a script nobody measured takes
  the slowest: over-admitting text past a model's audio ceiling truncates,
  under-admitting only splits. This is also what the field does — espeak-ng,
  ICU and Wyoming's sentence splitter all key on the character and take no
  language; Thai, which UAX #29 gives no terminator, splits on length there
  as here.
- **A word a model misreads is the model's to declare, and the stand-in the
  language's.** `ModelSpec.misreads` names the words (Hojo 40M: 行程, read as
  héng chéng); `text/zh/standins.tsv` says what to respell each as (形程),
  and only the declaring model gets the respelling. A declaration without a
  stand-in fails the catalog tests.
- **Whether a bare digit survives is the model's to declare.** Hojo 40M sets
  `needs_number_words` (an unexpanded digit is silent or replaced, upstream
  issue Hojo-TTS-Light#7), so bare numbers are expanded for it by default. No model here declares `reads_numerals`, so unit, time and date
  normalisation stays on for every language and the integration must not
  gate it on the tag.

## Consequences

- Adding a language is a locale package beside `zh/` and `en/`, registered in
  `text/pipeline.py`; adding a pass is a pattern in `passes.py`, a render per
  script and a row in that script's `_PASSES` tuple in the position the order
  requires. Neither touches the API, the UI or the integration.
- The test for where a new fact belongs: if the answer depends on which
  character it is, it is a row in `text/scripts.py`; if it depends on what
  language the reader speaks, it is the locale's. A language whose script is
  already in the table is read and split correctly before its locale exists.
- `tests/test_text.py` and `tests/test_english.py` pin the readings; they
  are the cheapest regression the project has and the most valuable.
