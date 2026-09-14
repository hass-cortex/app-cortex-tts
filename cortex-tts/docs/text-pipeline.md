# The text pipeline

None of the [models](models.md) has a text front-end. Each pronounces the
glyphs it is given, and two things they are routinely given are glyphs they
cannot pronounce. The app rewrites both before synthesis, and for Chinese that
rewrite is the difference between a voice and noise.

## Traditional Chinese

The tokenizer covers Traditional characters, so nothing fails — the model
simply produces the wrong sounds. Measured over 14 Home-Assistant-shaped
sentences, scored by transcribing the output and comparing it to the input:

| Input                                  | Character error rate |
| -------------------------------------- | -------------------- |
| Human recording (measurement floor)    | ~0%                  |
| 40M, Traditional text as-is            | 32.4%                |
| 80M clone, Traditional text as-is      | 39.1%                |
| **40M, converted to Simplified first** | **4.4%**             |

At 32% the content words are gone: 客廳的燈 comes out as something like
「确定的的」. The app converts Traditional glyphs to Simplified before synthesis,
using OpenCC's `t2s` — a **glyph-only** conversion. The phrase-aware variants
would rewrite 設定 to 设置 and 訊號 to 信号, changing the words spoken aloud;
they score no better, so the one that preserves Taiwanese wording is used.

## Taiwan readings

Converting the script makes the words recognisable; it does not make them
Taiwanese. The models are trained on mainland speech, so a word Taiwan reads
differently comes out in the mainland reading: 垃圾 as lā jī rather than lè sè,
企業 as qǐ yè rather than qì yè, 星期 as xīng qī rather than xīng qí. There
is no setting for it, and the cloned voice does not help: a Taiwanese
reference gives the voice a Taiwanese accent, not Taiwanese readings.

The models do take a phonetic hint — OmniVoice reads `LE4SE4`, Qwen3-TTS reads
`le4 se4` — but measured, it is fragile: a hint at the start of a sentence is
spelled out letter by letter, and four in one sentence add syllables that were
never written. Hojo and MOSS read neither form at all. What every model reads
reliably is an ordinary character, so the app respells the word with a
**homophone**: a character that has the Taiwan reading and no other. 垃圾
becomes 乐色, 企業 becomes 气业, 微波爐 becomes 维坡炉. The model is asked to
say a nonsense word, and says it correctly — measured on all four model
families with four respellings in one sentence and one at the start, every
output transcribing back to the intended words.

The table is generated, not written: `scripts/taiwan_readings.py` takes the
McBopomofo input method's dictionary (MIT; 140k words with their Taiwan
readings), compares each word's reading with the mainland reading of its
Simplified form, and keeps the ones that differ in more than tone sandhi.
Each differing syllable gets the most common character that both sides read
that way by default. A word whose reading is contextual (長, 當, 重) is kept
only when the dictionary's own default settles it, and a word for which no
clean stand-in exists (熟悉 shóu has none) is left as the model would read it.
The result is 5,237 words in `src/cortex_speech/text/zh/taiwan_readings.tsv`,
keyed by the Simplified form because the pass runs after script conversion;
each rewrite is listed in the **Rewrites** ledger as `reading`.

Japanese never meets it. Kanji share glyphs with the table (研究) and none of
its readings, and the pass is Chinese's: a `ja` request has no such pass, and
an untagged text with kana in it is read as `ja`.

## Numbers, units and times

`26.5°C`, `68%`, `14:35` and `2026-09-06` are read as noise — 36–50% character
error rate on their own. Home Assistant emits these constantly, so the app
expands them into words before synthesis. What the model receives, exactly as
`prepare()` returns it — Simplified, because script conversion runs after:

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

**The language of the words is the request's `language`.** It picks the
locale — Chinese and English have their own, with the unit and date tables
above; any other language gets its numbers read by `num2words` (some forty
languages), its unit symbols named the way CLDR names them in that language
(`26.5°C` → "sechsundzwanzig Komma fünf Grad Celsius"; `%`, `kWh`, `km/h`,
`hPa` and the rest of the table above), an ISO date laid out the way CLDR
lays it out with its numbers in words (`2026-09-14` → "vierzehnte September
zweitausendsechsundzwanzig", "2026年9月14日"), and a clock literal as hour
words then minute words (`14:35` → "vierzehn fünfunddreißig") — so a German
sentence never gets English words. A decimal comma is read where the
language writes one (`26,5`), while Home Assistant's own `26.5` and `1,234`
keep their machine meaning everywhere. Only those fixed shapes: this locale
parses no more than it is sure of, and a number it is not sure of stays as
digits, because a number read wrong is worse than one left unread. An "Uhr"
between the hour and the minute, or 時 and 分, is what a language's own
locale is for. A request with no tag uses the voice's language, and one with
neither is sniffed: kana is Japanese, 漢字 outnumbering Latin _words_ is
Chinese — tagged `zh-Hant` when written in Traditional glyphs — and anything
else is English. A bare `zh` is read the same way: Traditional glyphs make
it `zh-Hant`.

Whether a model could read the digits itself was measured rather than
assumed. It cannot: Qwen3-TTS, the one with a language model behind it,
read "1,234 kWh" in German as "eins Komma zwei drei vierunddreißig Gb" and
Japanese digits as noise, so no catalog entry declares `reads_numerals` and
the expansion runs for every model. The flag exists so that a model which
does can be declared as such and have the generic locale stand aside — the
Chinese and English locales stay on regardless, having been measured against
every model and won.

**Only the fixed shapes are read by default.** A number with a unit, a
percent sign, a clock colon or an ISO date around it says what it is; a
number on its own does not. `撥打 110`, `電話 0912345678`, `302號房`,
`RTX 4090`, `John 3:16`, `50-50` and `COVID-19` were all read as quantities
by the earlier rule — one hundred and ten, nine hundred million, a range,
minus nineteen — and a wrong reading misleads where digits left alone
merely go unread. So a bare number stays as digits unless the request says
`expand_numbers: true`, which a caller may do when it knows its numbers are
counts (a template that formats sensor values without units) — or, better,
writes them as words itself. The rule is the same in every locale.

With bare numbers on, a number welded after a letter is an identifier and is
read digit by digit (`P0` → "P zero"); one before a letter carries a unit and
stays a quantity (`24V` → "twenty-four V"). A range is read with its unit
once (`25-30°C` → 攝氏二十五到三十度) whether or not bare numbers are on;
thousands separators are dropped rather than read as pauses.

Latin words inside a Chinese sentence are left alone — the models read English
natively, so `Home Assistant` and `Roborock` pass through untouched. A bare
English sentence is not converted either: `t2s` has nothing to do to it.

The unit table is fixed (`SUFFIX_UNITS` in `src/cortex_speech/text/zh/normalize.py`);
an unusual unit passes through unexpanded until it is added there.

## Order, and the stop at the end

Normalisation runs before script conversion, because it emits Traditional
number words and only the conversion pass can make them pronounceable. Within
a normaliser, a construct claims its number before the bare-number pass reads
that digit as a quantity: `68%` is a percentage before `68` is a count.

The prepared text is then split into segments of at most 120 characters, and
**every segment ends in sentence-final punctuation**. The model stops only when
it samples an end-of-speech token; without a stop in the text it misses the
cue and invents a syllable. Measured: `客廳的燈已經打開了` produced a stray
「哈」, the same text with a full stop did not. A caller need not end a message
in punctuation — the app adds it.

## The switches

`normalize_text` is every language's, on by default; `expand_numbers` is
every language's too, off by default (see above). `convert_script` and
`taiwan_readings` are Chinese's alone: left out of a request, the language
decides them — conversion is always on, readings are on for `zh-TW` and
`zh-Hant` (a bare `zh` counts when the text is Traditional) and off for
`zh-CN` — and a request may still say `true` or `false` outright, though
readings cannot run with conversion off, being keyed by the Simplified form. On `/api/speak` they are fields; in a `tts.speak`
call they are the same names under `options:`. `/api/preview` answers with
`passes`: every switch the language has and what it was decided to be, which
is what the UI's chips show — and why the two Chinese chips are not there for
a German text.

The first two exist for a caller whose text is already prepared, not as
preferences: turning conversion off for ordinary Traditional Chinese makes the
voice unintelligible, and turning normalisation off leaves every digit silent.
The third is a preference — a listener who expects mainland readings turns it
off, or sends `zh-CN`.

## Seeing it before hearing it

The app's UI shows the prepared text — **What the model is asked to say** —
beside the composer, so you can read it before spending a synthesis on it.

![The composer and the prepared text](https://raw.githubusercontent.com/hass-cortex/app-cortex-tts/main/images/composer.png)

Left is what you typed; right is what the model receives. Each row under
**Rewrites** is one change: `number` for an expansion, `script` for the glyph
conversion with the count of glyphs it touched, `reading` for a word respelled
for its Taiwan reading, `stop` for punctuation added at the end. A pass that did not fire leaves no row — which is how you tell "nothing
needed rewriting" from "the switch is off". `POST /api/preview` returns the
same thing without loading a model.
