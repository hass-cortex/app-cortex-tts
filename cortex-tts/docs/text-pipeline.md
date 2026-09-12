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

**The language of the words follows the text around them, not a setting.** A
sentence that reads as Chinese — 漢字 outnumbering Latin _words_ — gets Chinese
numbers; one that reads as English gets English ones. A request carries flags,
never a language, so the text itself is the only thing that can answer.

A number welded after a letter is an identifier and is read digit by digit
(`P0` → "P zero"); one before a letter carries a unit and stays a quantity
(`24V` → "twenty-four V"). A range is read with its unit once (`25-30°C` →
攝氏二十五到三十度), and thousands separators are dropped rather than read as
pauses.

Latin words inside a Chinese sentence are left alone — the models read English
natively, so `Home Assistant` and `Roborock` pass through untouched. A bare
English sentence is not converted either: `t2s` has nothing to do to it.

The unit table is fixed (`SUFFIX_UNITS` in `src/cortex_speech/text/normalize.py`);
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

## The two switches

Both passes are on by default and can be turned off per request:
`normalize_text` and `convert_script` on `/api/speak`, and the same names under
`options:` in a `tts.speak` call. They exist for a caller whose text is already
prepared, not as preferences: turning conversion off for ordinary Traditional
Chinese makes the voice unintelligible, and turning normalisation off leaves
every digit silent.

The Home Assistant integration turns normalisation on for every language —
the model pronounces no numeral in any script — and script conversion on
whenever the pipeline language is Chinese.

## Seeing it before hearing it

The app's UI shows the prepared text — **What the model is asked to say** —
beside the composer, so you can read it before spending a synthesis on it.

![The composer and the prepared text](https://raw.githubusercontent.com/hass-cortex/app-cortex-tts/main/images/composer.png)

Left is what you typed; right is what the model receives. Each row under
**Rewrites** is one change: `number` for an expansion, `script` for the glyph
conversion with the count of glyphs it touched, `stop` for punctuation added at
the end. A pass that did not fire leaves no row — which is how you tell "nothing
needed rewriting" from "the switch is off". `POST /api/preview` returns the
same thing without loading a model.
