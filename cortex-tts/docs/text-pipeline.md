# The text pipeline

None of the [models](models.md) has a text front-end. Each pronounces the
glyphs it is given, and two things they are routinely given are glyphs they
cannot pronounce. The app rewrites both before synthesis; for Chinese that
rewrite is the difference between a voice and noise.

## Traditional Chinese

The tokenizer covers Traditional characters, so nothing fails — the model
produces the wrong sounds. Measured over 14 Home-Assistant-shaped sentences,
scored by transcribing the output and comparing it to the input:

| Input                                  | Character error rate |
| -------------------------------------- | -------------------- |
| Human recording (measurement floor)    | ~0%                  |
| 40M, Traditional text as-is            | 32.4%                |
| **40M, converted to Simplified first** | **4.4%**             |

At 32% the content words are gone: 客廳的燈 comes out as 「确定的的」. The app
converts Traditional glyphs to Simplified before synthesis with OpenCC's
`t2s`, a **glyph-only** conversion. The phrase-aware variants would rewrite
設定 to 设置 and 訊號 to 信号, changing the words spoken, and score no better.

## Taiwan readings

Converting the script makes the words recognisable, not Taiwanese. The models
are trained on mainland speech, so 垃圾 comes out as lā jī rather than lè sè,
企業 as qǐ yè rather than qì yè, 星期 as xīng qī rather than xīng qí. A
Taiwanese reference gives a clone a Taiwanese accent, not Taiwanese readings.

OmniVoice takes a phonetic hint (`LE4SE4`), but measured, it is fragile: a
hint at the start of a sentence is spelled out letter by letter, four in one
sentence add syllables never written, and Hojo and MOSS read no hint at all.
What every model reads reliably is an ordinary character, so the app respells
the word with a **homophone** that has the Taiwan reading and no other: 垃圾 →
乐色, 企業 → 气业, 微波爐 → 维坡炉. Measured on every model family with four
respellings in one sentence and one at the start, every output transcribed
back to the intended words.

The table is generated, never written: `scripts/taiwan_readings.py` takes the
McBopomofo input method's dictionary (MIT; 140k words with their Taiwan
readings), compares each word's reading with the mainland reading of its
Simplified form, and keeps the ones that differ in more than tone sandhi. Each
differing syllable gets the most common character both sides read that way by
default. A word whose reading is contextual (長, 當, 重) is kept only when the
dictionary's own default settles it, and one with no clean stand-in (熟悉
shóu) is left as the model would read it; the script prints which reason left
a word out. The result is 5,214 words in
`src/cortex_speech/text/zh/taiwan_readings.tsv`, keyed by the Simplified form
because the pass runs after script conversion; each rewrite is a `reading` row
in the **Rewrites** ledger.

A word one model reads with the wrong character is handled the same way but
declared differently. 行程 is xíng chéng everywhere, yet Hojo 40M reads it as
héng chéng, so its catalog entry declares `misreads=("行程",)` and the `zh`
locale respells the word from a hand-kept table, `standins.tsv` (行程 →
形程), for that model alone — MOSS and OmniVoice read the word right and are
handed it untouched. The defect is the model's to declare and the fix is the
language's; a declaration without a stand-in fails the catalog tests. The
boundary check applies here too, so 银行程序 keeps its 行. `/api/preview`
lists these respellings beside the Taiwan readings, so the ledger shows one
for Hojo 40M and none for the others.

### Which occurrences it fires on

The table is keyed by words and the text is not segmented, so a match is not
yet evidence that those characters are a word _in this sentence_. `在为` is an
entry (在為, zài wéi) and `正在为你查询` contains it, so the substituter took
it and the model read wéi where the sentence says wèi — 在 belongs to 正在.
Measured over 15 ordinary sentences, 10 were rewritten and 4 of those were
wrong, all the commonest shape an Assist reply takes (正在為您…, 現在為您…,
這只是…).

Dropping entries does not fix it: **5,200 of the 5,214 can straddle a boundary
in principle**, and the ones that do are neither rare (`在为` occurs 100 times
in the corpus, `夕阳` 116) nor weakly bound — by pointwise mutual information
`理发` (−0.78), `长发` (−0.66) and `法子` (−0.55) are words worth rewriting and
sit below artefacts like `为对` (−0.99) and `而为` (−0.90). Whether a span is
a word is a property of the occurrence, so it is decided there.

`readings._straddles` offers each edge of a match to its neighbour: if the
character before it forms a word with the match's first character, or the one
after with its last, and that word is at least as common as the match, the
match loses the character. The words and counts are `boundary_words.tsv`, from
the same generator and corpus — 53,521 of them, every two-character word that
can reach an entry's edge. Counts make it an arbitration rather than a veto:
中用 could claim the 中 of `不中用`, but it is rarer than the entry, so the
entry keeps it. Where the comparison goes the other way the pass does not
fire, the safe direction — an unrewritten word keeps the mainland reading and
is understood; a wrongly rewritten one is a different word.

Japanese never meets it: kanji share glyphs with the table (研究) and none of
its readings. A `ja` request has no such pass, and untagged text with kana in
it is read as `ja`.

## Numbers, units and times

`26.5°C`, `68%`, `14:35` and `2026-09-06` are read as noise — 36–50% character
error rate on their own — and Home Assistant emits them constantly, so the app
expands them into words before synthesis. What the model receives, exactly as
`prepare()` returns it (Simplified, because script conversion runs after):

| Input                       | What the model is asked to say                                           |
| --------------------------- | ------------------------------------------------------------------------ |
| `目前 26.5°C`               | 目前摄氏二十六点五度。                                                   |
| `濕度 68%`                  | 湿度百分之六十八。                                                       |
| `現在 14:35`                | 现在十四点三十五分。                                                     |
| `今天是 2026-09-06`         | 今天是二零二六年九月六日。                                               |
| `今天用了 3.2 kWh`          | 今天用了三点二度电。                                                     |
| `溫度 22–26°C`              | 温度摄氏二十二到二十六度。                                               |
| `總共 12,345 元`            | 总共一万二千三百四十五元。                                               |
| `It is 26.5°C and 68% now.` | It is twenty-six point five degrees Celsius and sixty-eight percent now. |

**The language of the words is the request's `language`.** Chinese and English
have written locales; any other language gets the generic one — numbers from
`num2words` (some forty languages), unit symbols and ISO dates as CLDR names
and lays them out in that language, a clock literal as hour words then minute
words (`26.5°C` → "sechsundzwanzig Komma fünf Grad Celsius", `14:35` →
"vierzehn fünfunddreißig") — so a German sentence never gets English words. A
decimal comma is read where the language writes one (`26,5`), while Home
Assistant's own `26.5` and `1,234` keep their machine meaning everywhere. The
generic locale parses only those fixed shapes and leaves a number it is not
sure of as digits, because a number read wrong is worse than one left unread;
an "Uhr" between hour and minute, or 時 and 分, is what a written locale is
for. A request with no tag uses the voice's language; one with neither is
sniffed — kana is Japanese, 漢字 outnumbering Latin _words_ is Chinese
(`zh-Hant` in Traditional glyphs, a bare `zh` read the same way), anything
else English.

No model reads the digits itself: no catalog entry declares `reads_numerals`,
so the expansion runs for every model. The flag lets a model that does be
declared, with the generic locale standing aside while Chinese and English
stay on.

**Only the fixed shapes are read by default.** A number with a unit, a percent
sign, a clock colon or an ISO date around it says what it is; a number on its
own does not. In Chinese the unit may be a word as well as a symbol
(`25.9 度`, `3 分鐘`, `2 個月`; `WORD_UNITS`, both scripts), a four-digit
number before 年 is a year read digit by digit, and a number before 號 is a
label. Read as quantities, `撥打 110`, `電話 0912345678`, `302號房`,
`RTX 4090`, `John 3:16`, `50-50` and `COVID-19` become one hundred and ten,
nine hundred million, a range, minus nineteen — a wrong reading misleads where
digits left alone merely go unread. So a bare number stays as digits unless
the request says `expand_numbers: true` (a template formatting sensor values
without units), or, better, the caller writes it as words. The rule is the
same in every locale, with one exception: a model that cannot say a digit at
all. Measured with `撥打 110，房間在 302 號房，型號 RTX 4090，共 25 人` and its
English counterpart, MOSS read every number itself (110 as 一幺幺零, the phone
way), OmniVoice read them too, but Hojo produced 十億億安 and,
in English, nonsense. For such a model — `needs_number_words` in the catalog,
reported by `/api/models` — bare numbers are expanded by default;
`expand_numbers` in the request still overrides it either way.

With bare numbers on, a number welded after a letter is an identifier read
digit by digit (`P0` → "P zero"). Whether or not they are on, a number before a
letter the unit table knows is a quantity with its unit read out (`24V` →
"twenty-four volts"), a range is read with its unit once (`25-30°C` →
攝氏二十五到三十度), and thousands separators are dropped rather than read as
pauses. Latin words inside a Chinese sentence pass through untouched — the
models read English natively — and a bare English sentence is not converted.

The unit symbols are the ones Home Assistant's `homeassistant/const.py`
defines (`UnitOf*`: `W`, `kWh`, `hPa`, `L/min`, …). English and the generic
locale name them through CLDR (`src/cortex_speech/text/units.py`, via `babel`:
"forty-eight watts", "zwei Liter pro Minute"), with the number's agreement; a
symbol CLDR has no name for in that language stays as its letters after the
number words (`5 ppm` → "fünf ppm" in German). Chinese has its own table
(`SUFFIX_UNITS` in `text/zh/normalize.py`: `kWh` → 度電, `MHz` → 兆赫) and
`WORD_UNITS` for unit words. Symbols that are also words or labels (`in`,
`st`, `ac`, `ha`, `K`, `B`, `d`, `w`, `y`) are left out on purpose.

## Order, and the stop at the end

Normalisation runs before script conversion, because it emits Traditional
number words and only the conversion pass can make them pronounceable. Within
a normaliser, a construct claims its number before the bare-number pass reads
that digit as a quantity: `68%` is a percentage before `68` is a count.

The prepared text is then split into segments the chosen model can say whole.
Where a sentence ends is the character's to say, not the language's:
`text/scripts.py` lists the terminators once — the ASCII and CJK sets and the
Unicode `STerm` marks of the other scripts, 「।」「؟」「።」 among them — and the
splitter, the live sentence buffer and the terminator pass all derive from
that list, so an untagged request and a reply arriving in pieces cut in the
same places. Thai has no terminator in Unicode and splits on length alone.
`ModelSpec.segment_limit` turns the model's own audio ceiling into a length of
text with the speech rate of the script; a script nobody measured is assumed
as slow as the slowest measured one, so it is split more, never truncated. A
model that establishes no ceiling is split on sentence boundaries alone. The figures are per model in
[Models](models.md#how-much-text-one-synthesis-takes). **Every segment ends in
sentence-final punctuation.** The model stops only when it samples an
end-of-speech token; without a stop it misses the cue and invents a syllable —
`客廳的燈已經打開了` produced a stray 「哈」, the same text with a full stop did
not. A caller need not end a message in punctuation; the app adds it.

## The switches

`normalize_text` is every language's, on by default. `expand_numbers` is every
language's too; left out, it is the model's call — on only for one that cannot
say a digit. `convert_script` and `taiwan_readings` are Chinese's alone: left
out, the language decides them — conversion always on, readings on for `zh-TW`
and `zh-Hant` (a bare `zh` counts when the text is Traditional) and off for
`zh-CN` — and a request may still say `true` or `false` outright, though
readings cannot run with conversion off, being keyed by the Simplified form.
Between the request and those answers sit the settings: a rule per model and
language (`text_rules`, on the app's Settings page and over
[`PUT /api/settings`](api.md#settings-over-the-api)) answers any switch the
request left out, so a household that wants bare numbers read on MOSS in
Chinese sets that once. There is no per-voice rule.

On the wire the switches are fields of a reply's opening frame; in a
`tts.speak` call they are the same names under `options:`. `/api/preview`
answers with `passes` — every switch the language has and what it was decided
to be — which is what the UI's chips show, and why the two Chinese chips are
absent for a German text.

`normalize_text` and `convert_script` exist for a caller whose text is already
prepared, not as preferences: conversion off for ordinary Traditional Chinese
makes the voice unintelligible, normalisation off leaves every digit silent.
`taiwan_readings` is a preference — a listener who expects mainland readings
turns it off, or sends `zh-CN`.

## Seeing it before hearing it

The app's UI shows the prepared text — **What the model is asked to say** —
beside the composer, so you can read it before spending a synthesis on it.

![The composer and the prepared text](https://raw.githubusercontent.com/hass-cortex/app-cortex-tts/main/images/composer.png)

Each row under **Rewrites** is one change: `number` for an expansion, `script`
for the glyph conversion with the count of glyphs it touched, `reading` for a
Taiwan respelling, `stop` for punctuation added at the end, `text` for a
rewrite carrying no digit. A pass that did not fire leaves no row, which is
how "nothing needed rewriting" is told from "the switch is off".
`POST /api/preview` returns the same thing without loading a model.
