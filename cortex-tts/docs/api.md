# HTTP API

The app is usable on its own, not only through the Home Assistant integration.
Full OpenAPI, with every endpoint and an in-browser console, is served at
`/api/docs`.

## Reaching it

The port is **not published by default**: the integration reaches the app over
the Supervisor's internal network as `http://local-cortex-tts:8771`, and the
admin UI comes through ingress. To call it from elsewhere on your network,
publish port 8771 under the app's **Network** settings; the address is then
`http://<home-assistant-host>:8771`.

## Authentication

Everything under `/api` and `/v1` takes the key from the app's
`discovery_api_key` option, as `Authorization: Bearer <key>` or an `X-API-Key`
header. The three FastAPI-generated pages — `/api/docs`, `/api/openapi.json`
and `/redoc` — are registered on the app rather than on the router, so they
answer without one. Requests arriving through ingress are already authenticated by Home
Assistant and skip the check; ingress is recognised by the Supervisor's
`X-Ingress-Path` header **from the Supervisor's own address**, so the header
alone, from anywhere else, proves nothing. `/health` never needs a key. An
empty configured key disables the check, which is only sane while the port
stays unpublished.

`/api/speak/live` takes one more form, for browsers only: a `WebSocket`
constructor cannot set a header, so the key may be offered as the second
entry of the handshake's subprotocol list, after `cortex-tts` —
`new WebSocket(url, ["cortex-tts", key])`. It travels in the same handshake
the header would have. Anything that can set headers still does.

## Endpoints

| Method | Path                         | Purpose                                                                     |
| ------ | ---------------------------- | --------------------------------------------------------------------------- |
| GET    | `/health`                    | liveness; version, `api_version`, resident count, what is loading, provider |
| GET    | `/api/defaults`              | the configured default model and voice                                      |
| GET    | `/api/settings`              | every stored setting, as it is now in force                                 |
| PUT    | `/api/settings`              | change some of them; omitted fields keep their value                        |
| GET    | `/api/models`                | catalog + per-model state (downloaded/loaded/progress)                      |
| POST   | `/api/models/{id}/download`  | start a download; poll `/api/models`                                        |
| DELETE | `/api/models/{id}`           | remove the bundle from disk; 409 while it is downloading                    |
| POST   | `/api/models/{id}/load`      | make it resident                                                            |
| POST   | `/api/models/{id}/unload`    | evict it                                                                    |
| GET    | `/api/voices`                | voices across downloaded models, or one model's (`?model=`)                 |
| POST   | `/api/preview`               | run the text path only; no model is loaded                                  |
| WS     | `/api/speak/live`            | a reply spoken while it is still being written; the server paces it         |
| POST   | `/v1/audio/speech`           | synthesise to a finished file, OpenAI-shaped                                |
| GET    | `/api/references`            | cloned-voice reference recordings                                           |
| POST   | `/api/references`            | add one (multipart: audio + transcript + metadata)                          |
| PATCH  | `/api/references/{id}`       | correct the name, a transcript, the gender label and/or the language        |
| DELETE | `/api/references/{id}`       | remove it, and the voice it defined                                         |
| GET    | `/api/references/{id}/audio` | play the recording back                                                     |
| DELETE | `/api/models/{id}/stats`     | forget what this host measured for one model; returns it                    |
| DELETE | `/api/stats`                 | the same for every model; returns the catalog                               |

## What this host measured

`GET /api/models` carries an `rtf` list per model, one entry per cost line:
one covering the model's own voices, whose cost differs by 4%, and one per
cloned voice, whose recording rejoins the prompt on every synthesis and so
costs in proportion to its own length ([Delivering a reply](delivery.md) measures
it). Each entry names its `kind`
(`builtin`, `designed` or `reference`) and, on a clone, the `voice` it was
measured with. The rest is the render model fitted to the requests this host
has actually served: `per_audio` (render seconds per audio second, the figure
a card shows), `fixed_s` (what a request costs before any audio), `spread_s`,
`cjk_per_s`, `latin_per_s` and `requests`.

The list is empty until three requests have gone into that cost line, because
two points are not a line — three renders across any of a model's own voices,
or three of one clone. `per_audio` rather than an average of what each
request cost: a reply the server planned into ten short requests would otherwise
read as ten slow ones, each charged the whole fixed cost.

`DELETE /api/models/{id}/stats` forgets one model's and `DELETE /api/stats`
every model's — for when the host changed under them and the stored figures
describe a machine that is gone.

## Speaking

Every reply is spoken over `/api/speak/live`, whose opening frame settles
what follows. These are its fields, and `/api/preview` takes the text ones so
a caller can see what the model would be asked to say without loading it.

| Field             | Default              | Meaning                                                                                                                                                                      |
| ----------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `text`            | required             | Up to 4000 characters, in whatever script you write                                                                                                                          |
| `model`           | the default          | A model id from `/api/models`                                                                                                                                                |
| `voice`           | the default          | A voice id the model offers; the first available when it does not                                                                                                            |
| `format`          | `mp3`                | `mp3`, or `wav` for raw PCM. A stream cannot declare a length, so the file containers are `/v1/audio/speech`'s                                                               |
| `normalize_text`  | `true`               | Expand numbers, units, dates and clock literals ([why](text-pipeline.md))                                                                                                    |
| `expand_numbers`  | model decides        | Also read a bare number — no unit, clock or date around it — as a quantity. Left out: on for a model that cannot say a digit (Hojo), off otherwise ([why](text-pipeline.md)) |
| `convert_script`  | the language decides | Chinese only: Traditional → Simplified glyph conversion                                                                                                                      |
| `taiwan_readings` | the language decides | Chinese only: respell words Taiwan reads differently ([why](text-pipeline.md))                                                                                               |
| `temperature`     | the setting          | Sampling temperature 0–1, for models that have one                                                                                                                           |
| `language`        | the voice's          | The language of the text, as a whole tag: picks how it is prepared on every model, and which language the model reads it in on those that take one                           |
| `instruct`        | none                 | A plain-language instruction beside the voice, for the one model that does                                                                                                   |

A live reply reports its measurements in frames rather than headers — see
`rendered` and `done` below. `/v1/audio/speech`, which answers a finished
file, carries them as `X-Cortex-Model`, `X-Cortex-Voice`,
`X-Cortex-Inference-Ms`, `X-Cortex-Audio-Seconds`, `X-Cortex-Rtf` and
`X-Cortex-Segments`.

### Speaking live

`/api/speak/live` is a WebSocket, one per reply, for text that is still being
written — a conversation agent's answer arriving a few words at a time. The
client sends the words as they come; the server decides when to render what,
how much to hold back before the first sound, and whether to stream at all,
from what it has measured about the model on this host
([how](delivery.md)). Authenticate the handshake the same way as any request.

Client → server, JSON text frames:

| Frame    | Fields                                                                                                                                                                                                                                                                                                                                                                |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `start`  | first, once: `model`, `voice`, `format` (`mp3` default, `wav`), `mode` (`auto` default; `buffered`, `planned`, `unheld` and `streaming` insist on one delivery rather than letting the server choose, honoured as far as the reply allows — `done` says what actually happened), and the text switches, `temperature`, `language` and `instruct` from the table above |
| `text`   | `text`: the next piece of the reply, as written, up to 4000 characters a frame; any number of these                                                                                                                                                                                                                                                                   |
| `end`    | the reply is complete                                                                                                                                                                                                                                                                                                                                                 |
| `cancel` | stop: the listener is gone. Rendering stops at the engine's next checkpoint                                                                                                                                                                                                                                                                                           |

Server → client:

| Frame      | Fields                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ready`    | once the model and voice are settled and the engine is resident: `model`, `voice`, `bitrate`, `sample_rate`, `chunk_streaming`, and `mode` — `buffered` only when the caller asked for it, `streaming` otherwise; which planning it settles on is the first `batch` frame's to say                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `batch`    | before each request is rendered: `index` (from 1), `mode` (the plan in force: `streaming`, `planned`, `unheld` or `buffered`), `text` (what this request carries, as the writer wrote it) and `ends_sentence` (whether a gap after it would be heard as a pause between sentences rather than as broken), so a listener can show how the reply is being cut and delivered before its first audio                                                                                                                                                                                                                                                                                                                                                                                                                    |
| binary     | audio, in the requested container, in playback order; nothing else needs decoding. One frame carries at most 512 KB and a slice may fall anywhere, so concatenate the frames and decode the stream rather than the frame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `rendered` | after each request: `index`, `audio_s` and `render_ms` — what that request actually cost. `batch` can only carry the plan, and while audio is held back there are no bytes to read a cost from                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `done`     | last: `mode` as it was actually spoken: `whole` (one request, however it was planned — `whole` is about where the reply was cut, and one request has nowhere to cut), `streaming`, `planned`, or `buffered`, which survives the count because it is about releasing rather than cutting: one request heard as it rendered and the same one held to the end are twenty-two seconds apart, `batches` (how many requests the reply was rendered in), `audio_seconds`, `first_audio_ms`, `min_lead_s` (the least audio the listener held; negative means it ran dry), `render_ms` and `rtf` (what the model was busy for, and that over the audio — not what the listener waited), `load_ms` (making the model resident, paid before `ready`), `writer_ms` (from `ready` to `end`: how long the writer took), `wall_ms` |
| `error`    | `code` and `message`, then the socket closes: 1008 for a refusal before `ready`, 1011 for a failure after it. A refusal a route would also raise carries that route's code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

Closing the socket, sending `cancel`, or not reading for fifteen seconds all
stop the render. The request being rendered when that happens teaches the
model nothing; requests the same reply already finished were recorded as each
one landed. `POST /v1/audio/speech` is the other way round — it is one
request, so an abandoned one records nothing.

`POST /v1/audio/speech` is the OpenAI shape — `input`, `model`, `voice`,
`response_format` — so existing clients work unchanged. `response_format`
defaults to `wav` and accepts the same four formats; `opus`, `aac` and `pcm`
are refused.

`POST /api/preview` takes `text`, `model`, `language`, `normalize_text`,
`expand_numbers`, `convert_script` and `taiwan_readings` and returns `original`, `prepared`,
`segments`, `language` (the tag the text was read as, sniffed when none was
sent), `passes` (every switch that language has, and whether it ran) and
`readings` — the `{word, standin}` respellings applied, in text order —
without touching a model.

A reference recording's `language` (on `POST /api/references` and editable
with `PATCH /api/references/{id}`) is a whole tag as well: a voice labelled
`zh-TW` is Taiwanese, so text read in it gets Taiwan readings by default
even where the glyphs alone could not say — `zh` promises only Chinese.

`language` is sent whole — `zh-TW`, not `zh` (`zh_TW` is read as the
same tag). It is accepted on every model:
the text pipeline reads it on all of them, and a model that takes a language
is told it too. How much of a tag means anything is the model's to decide:
Qwen3-TTS names two Chinese dialects apart from Chinese, OmniVoice names 646
languages including Cantonese, and reducing the tag in the caller would throw
that away before either got to say it mattered. Left out, the voice's own
language is used, and failing that the text is sniffed.
The engine tries the whole tag, then its shorter forms, and refuses what it
does not read. Which models take `language` and `instruct` at all is
`language_choice` and `style_instruction` in `/api/models`; a model that
declares neither refuses the field rather than accepting it and doing nothing
with it, because an ignored field is indistinguishable from a working one.

## Why a stream is MP3

`/api/speak/live` answers MP3: a bare sequence of self-describing frames, with
no container, no length field and no index, which is the only honest thing to
send when the length is not known yet. A WAV stream has to declare a length
before the audio exists, and a general-purpose player given the maximal one
waits for a file it believes is six hours long. FLAC and OGG both need a size
or an index written before the audio exists, so they are not offered. The
bitrate — `bitrate` in the `ready` frame — is the one measurement that exists
before the first sample, and it is what turns a byte count into a duration
downstream.

## Errors

Every error, whatever raised it, is a JSON body of `{"code", "message"}` —
a route's own refusal, an unknown path, and a request body pydantic rejected
(422) alike.

| Status | Code                   | When                                                                             |
| ------ | ---------------------- | -------------------------------------------------------------------------------- |
| 400    | `EMPTY_TEXT`           | Nothing left to say once punctuation was stripped                                |
| 400    | `NO_TEMPERATURE`       | A temperature for a model that has none (MOSS, OmniVoice)                        |
| 400    | `NO_STYLE_INSTRUCTION` | An `instruct` for a model that takes none                                        |
| 400    | `UNSUPPORTED_LANGUAGE` | A `language` the model does not read                                             |
| 400    | `BAD_REFERENCE`        | Unreadable audio, wrong length, an empty transcript, or a recording cut mid-word |
| 401    | `AUTH_REQUIRED`        | No key, or the wrong one                                                         |
| 404    | `UNKNOWN_MODEL`        | No such model id                                                                 |
| 404    | `UNKNOWN_VOICE`        | The model does not offer that voice (ids are case-sensitive)                     |
| 404    | `UNKNOWN_REFERENCE`    | No such reference id                                                             |
| 409    | `MODEL_NOT_READY`      | The model is not downloaded                                                      |
| 409    | `NO_VOICE`             | The model has no voices yet — upload a reference                                 |
| 409    | `DOWNLOAD_RUNNING`     | Delete refused while the bundle is still arriving                                |
| 413    | `BAD_REFERENCE`        | An upload larger than any legal reference                                        |
| 422    | `VALIDATION`           | The request body failed validation; the message names the field                  |
| 422    | `NO_AUDIO`             | The model produced nothing for that text                                         |
| 499    | `ABANDONED`            | The caller went away before the render finished                                  |
| 500    | `ENGINE_ERROR`         | The model failed in a way not listed above                                       |
| 503    | `OUT_OF_MEMORY`        | The card had no memory left; the engine holding it is dropped                    |
| 500    | `SETTINGS_NOT_WRITTEN` | The settings file could not be stored                                            |
| 503    | `PROVIDER_UNAVAILABLE` | `cuda` was required and did not answer                                           |

## Settings over the API

`GET /api/settings` returns the stored settings as they are in force.
`PUT /api/settings` takes any subset; a field that fails validation keeps its
previous value rather than rejecting the form, and the reply names it under
`ignored` beside `reloaded`, which says whether resident models were dropped
to adopt a thread count or execution provider. See the App Store page for what
each setting does.

`text_rules` is what the four text switches default to when a request leaves
them out, per model and language: a list of `{model, language, normalize_text,
expand_numbers, convert_script, taiwan_readings}` where `model` is a catalog id
or null for every model, `language` a tag the request's resolved language must
equal or extend (`zh` covers `zh-TW`) or null for every language, and each
switch `true`, `false` or null for "the pipeline's call". Rules cascade per
switch, the most specific one that says something winning — model and language
over either alone, either over neither; equal ones, the later. Sending the
list replaces it whole. A rule naming a model the catalog lacks refuses the
whole list, reported under `ignored`.

## Versioning

`GET /health` reports `api_version`, bumped when a route, field or header a
client reads changes shape; the app's release `version`, reported beside it,
says nothing about the wire. A client compares the number before trusting
anything else it reads. `api_version` is 4.

A field a client did not know about is not a change of shape, so adding one
does not bump it — the integration refuses to set up on a mismatch, and
widening the response is not a reason to stop an install that works.
`/health` gained `loading_models` that way: it lists what is being built right
now, at most one, and is how `loaded_models: 0` during a load is told apart
from `loaded_models: 0` because nothing is happening.

## For integration authors

On Home Assistant OS the app announces itself through Supervisor discovery
with `{host, port, api_key}` under the service `cortex_tts`; the key is
generated on first start. Whenever the set of voices changes — a model
downloaded or deleted, a reference added or removed — the app fires
`cortex_tts_models_changed` on the Home Assistant event bus, which is how the
integration adds and removes entities without a reload.
