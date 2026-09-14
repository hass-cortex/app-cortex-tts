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
header. Requests arriving through ingress are already authenticated by Home
Assistant and skip the check; ingress is recognised by the Supervisor's
`X-Ingress-Path` header **from the Supervisor's own address**, so the header
alone, from anywhere else, proves nothing. `/health` never needs a key. An
empty configured key disables the check, which is only sane while the port
stays unpublished.

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
| POST   | `/api/speak`                 | synthesise; the requested format (wav/flac/ogg/mp3) + `X-Cortex-*`          |
| POST   | `/api/speak/stream`          | the same, sent as it is produced (chunked MP3)                              |
| POST   | `/v1/audio/speech`           | the same, OpenAI-shaped                                                     |
| GET    | `/api/references`            | cloned-voice reference recordings                                           |
| POST   | `/api/references`            | add one (multipart: audio + transcript + metadata)                          |
| PATCH  | `/api/references/{id}`       | correct a transcript and/or the gender label                                |
| DELETE | `/api/references/{id}`       | remove it, and the voice it defined                                         |
| GET    | `/api/references/{id}/audio` | play the recording back                                                     |

## Speaking

`POST /api/speak` takes JSON:

| Field             | Default              | Meaning                                                                                                                                                                      |
| ----------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `text`            | required             | Up to 4000 characters, in whatever script you write                                                                                                                          |
| `model`           | the default          | A model id from `/api/models`                                                                                                                                                |
| `voice`           | the default          | A voice id the model offers; the first available when it does not                                                                                                            |
| `format`          | `wav`                | `wav`, `flac`, `ogg` or `mp3`                                                                                                                                                |
| `normalize_text`  | `true`               | Expand numbers, units, dates and clock literals ([why](text-pipeline.md))                                                                                                    |
| `expand_numbers`  | model decides        | Also read a bare number — no unit, clock or date around it — as a quantity. Left out: on for a model that cannot say a digit (Hojo), off otherwise ([why](text-pipeline.md)) |
| `convert_script`  | the language decides | Chinese only: Traditional → Simplified glyph conversion                                                                                                                      |
| `taiwan_readings` | the language decides | Chinese only: respell words Taiwan reads differently ([why](text-pipeline.md))                                                                                               |
| `normalize_level` | `true`               | Peak-normalise the finished waveform                                                                                                                                         |
| `temperature`     | the setting          | Sampling temperature 0–1, for models that have one                                                                                                                           |
| `language`        | the voice's          | The language of the text, as a whole tag: picks how it is prepared on every model, and which language the model reads it in on those that take one                           |
| `instruct`        | none                 | A plain-language instruction beside the voice, for the one model that does                                                                                                   |

The response is the audio, with the measurements in headers:
`X-Cortex-Model`, `X-Cortex-Voice`, `X-Cortex-Inference-Ms`,
`X-Cortex-Audio-Seconds`, `X-Cortex-Rtf`, `X-Cortex-Segments`. These are what
the integration's diagnostic sensors report for a buffered reply.

`POST /api/speak/stream` takes the same body and answers chunked audio as it is
produced — `mp3` by default, `wav` on request, `flac` and `ogg` refused because
a stream cannot declare a length ([why](streaming.md#why-a-stream-is-mp3)).
The headers are `X-Cortex-Model`, `X-Cortex-Voice`, `X-Cortex-Bitrate` (bits
per second, constant, so a byte count converts to a duration) and
`X-Cortex-Chunk-Streaming` (`1` when the model emits mid-sentence, `0` when
each request arrives whole). Everything that can fail — an unknown model or
voice, a model not downloaded, a model that will not load — fails before the
first byte; once the response has started the status is 200 and a failure can
only truncate the audio. Closing the connection early stops the rendering
behind it within a chunk. `normalize_level` is accepted and ignored: a stream
has no finished waveform to scale, and holds its chunks under one ceiling
instead.

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

## Errors

Every error, whatever raised it, is a JSON body of `{"code", "message"}` —
a route's own refusal, an unknown path, and a request body pydantic rejected
(422) alike.

| Status | Code                   | When                                                                             |
| ------ | ---------------------- | -------------------------------------------------------------------------------- |
| 400    | `EMPTY_TEXT`           | Nothing left to say once punctuation was stripped                                |
| 400    | `NO_TEMPERATURE`       | A temperature for a model that has none (MOSS, OmniVoice)                        |
| 400    | `NO_LANGUAGE_CHOICE`   | A `language` for a model whose voice decides it                                  |
| 400    | `NO_STYLE_INSTRUCTION` | An `instruct` for a model that takes none                                        |
| 400    | `UNSUPPORTED_LANGUAGE` | A `language` the model does not read                                             |
| 400    | `UNSUPPORTED_FORMAT`   | `flac` or `ogg` asked of the stream                                              |
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
| 500    | `ENGINE_ERROR`         | The model failed in a way not listed above                                       |
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
anything else it reads. `api_version` is 1.

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
