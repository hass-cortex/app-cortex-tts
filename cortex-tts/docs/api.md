# HTTP API

The app is usable on its own, not only through the Home Assistant integration.
The OpenAPI — every route, body and response, with an in-browser console — is
served at `/api/docs` and `/api/openapi.json`. Two routes need more than it
says: the WebSocket `/api/speak/live` is not in it at all
([Speaking live](#speaking-live)), and `POST /v1/audio/speech` returns headers
it does not describe ([Speaking](#speaking)).

## Reaching it

The port is **not published by default**: the integration reaches the app over
the Supervisor's internal network as `http://local-cortex-tts:8771`, and the
admin UI comes through ingress. To call it from elsewhere on your network,
publish port 8771 under the app's **Network** settings; the address is then
`http://<home-assistant-host>:8771`.

## Authentication

Everything under `/api` and `/v1` takes the key from the app's
`discovery_api_key` option, as `Authorization: Bearer <key>` or an `X-API-Key`
header. `/health` and the FastAPI-generated pages (`/api/docs`,
`/api/openapi.json`, `/redoc`) answer without one. Requests arriving through
ingress are already authenticated by Home Assistant and skip the check;
ingress is recognised by the Supervisor's `X-Ingress-Path` header **from the
Supervisor's own address**, so the header alone, from anywhere else, proves
nothing. An empty configured key disables the check, which is only sane while
the port stays unpublished.

`/api/speak/live` takes one more form, for browsers: a `WebSocket` constructor
cannot set a header, so the key may be offered as the second entry of the
handshake's subprotocol list, after `cortex-tts` —
`new WebSocket(url, ["cortex-tts", key])`.

## What this host measured

`GET /api/models` carries an `rtf` list per model, one entry per voice this
host has served: `kind` (`builtin`, `designed` or `reference`), `voice`, `rtf`
(render seconds over audio seconds, the median of the voice's most recent
requests on the execution provider in use), `samples` (how many requests the
median rests on, at most eight), `provider` (`cpu` or `cuda`) and `verdict`
(`streaming` or `buffered` — how a live reply in that voice is spoken under
`auto`; [Delivering a reply](delivery.md)). A cloned voice is its own entry
because its recording rejoins the prompt on every synthesis; nothing is pooled
across voices or borrowed from one. A voice is absent until three requests
have gone into it on the provider in use; a sample from the other provider
describes another machine and is not read.

`DELETE /api/models/{id}/stats` forgets one model's and `DELETE /api/stats`
every model's — for when the host changed under them, or after a benchmark
([Models](models.md#what-a-benchmark-cannot-measure)).

## Speaking

Every reply is spoken over `/api/speak/live`, whose opening frame settles what
follows; `POST /api/preview` takes the text fields and returns the prepared
text without loading a model; `POST /v1/audio/speech` is the OpenAI shape
(`input`, `model`, `voice`, `response_format`, default `wav`) for clients that
want a finished file. The fields are in the OpenAPI; what it does not say:

- `/v1/audio/speech` reports its measurements as headers — `X-Cortex-Model`,
  `X-Cortex-Voice`, `X-Cortex-Inference-Ms`, `X-Cortex-Audio-Seconds`,
  `X-Cortex-Rtf`, `X-Cortex-Segments`; a live reply reports the same in its
  `rendered` and `done` frames.
- A finished file is peak-normalised before it is encoded. A stream has no
  finished waveform to measure, so it carries a gain that only ever falls, far
  enough to keep each chunk under the same ceiling; a buffered live reply is
  held to its end and levelled like a file.
- `language` is a whole tag (`zh-TW`, not `zh`; `zh_TW` reads the same),
  accepted on every model: the text pipeline reads it on all of them, and a
  model that takes a language is told it too — the engine tries the whole tag,
  then its shorter forms, and refuses what it does not read. Left out, the
  voice's language is used, failing that the text is sniffed. Which models
  take `language` and `instruct` is `language_choice` and `style_instruction`
  in `/api/models`; a model that declares neither refuses the field, because
  an ignored field is indistinguishable from a working one. A reference
  recording's `language` is a whole tag as well: a voice labelled `zh-TW` gets
  Taiwan readings by default where the glyphs alone could not say.
- The four text switches are [the text pipeline](text-pipeline.md)'s;
  `/api/preview` answers with `passes` (every switch the language has, and
  whether it ran), `readings` (the `{word, standin}` respellings applied) and
  `language` (the tag the text was read as, sniffed when none was sent).

### Speaking live

`/api/speak/live` is a WebSocket, one per reply, for text that is still being
written. The client sends the words as they come; the server decides, from
what it has measured of the voice on this host, whether to speak the reply one
sentence at a time from a bank of audio or to hold it until it is rendered
([how](delivery.md)). Authenticate the handshake as any request.

Client → server, JSON text frames:

| Frame    | Fields                                                                                                                                                                                                                                                                                                                               |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `start`  | first, once: `model`, `voice`, `format` (`mp3` default, `wav`), `mode` (`auto` default: the verdict this host's measurement gives the voice; `streaming` and `buffered` insist on one outcome whatever it measured — `done` says which was spoken), and the text switches, `temperature`, `language` and `instruct` from the OpenAPI |
| `text`   | `text`: the next piece of the reply, as written, up to 4000 characters a frame; any number of these                                                                                                                                                                                                                                  |
| `end`    | the reply is complete                                                                                                                                                                                                                                                                                                                |
| `cancel` | stop: the listener is gone. Rendering stops at the engine's next checkpoint                                                                                                                                                                                                                                                          |

Server → client:

| Frame      | Fields                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ready`    | once the model and voice are settled and the engine is resident: `model`, `voice`, `bitrate`, `sample_rate`, `chunk_streaming`, `mode` (`streaming` or `buffered`, settled here and never revised), `rtf` (the voice's median on this host, or `null` while unmeasured) and `samples` (how many requests it rests on)                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `batch`    | before each request is rendered: `index` (from 1), `mode` (`streaming` or `buffered`) and `text` (what this request carries, as the writer wrote it — one sentence when streaming, every sentence that had arrived when buffered), so a listener can show how the reply is being cut and delivered before its first audio                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| binary     | audio, in the requested container, in playback order; nothing else needs decoding. One frame carries at most 512 KB and a slice may fall anywhere, so concatenate the frames and decode the stream rather than the frame                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `rendered` | after each request: `index`, `audio_s` and `render_ms` — what that request actually cost. `batch` can only carry the plan, and while audio is held back there are no bytes to read a cost from                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `done`     | last: `mode` as it was actually spoken, `streaming` or `buffered` — buffered is about releasing rather than cutting, so a one-request reply held to its end is buffered too; `batches` (how many requests the reply was rendered in), `audio_seconds`, `render_ms` and `rtf` (what the model was busy for, and that over the audio — not what the listener waited), `load_ms` (making the model resident, paid before `ready`), `writer_ms` (from `ready` to `end`: how long the writer took), `bank_wait_ms` (how long the first audio waited between being rendered and being released), `first_audio_ms`, `min_lead_s` (the least audio the listener held; negative means it ran dry by that much), `gap_at` (the request whose audio landed at that lowest lead) and `wall_ms` |
| `error`    | `code` and `message`, then the socket closes: 1008 for a refusal before `ready`, 1011 for a failure after it. A refusal a route would also raise carries that route's code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

Closing the socket, sending `cancel`, or not reading for fifteen seconds all
stop the render. The request being rendered when that happens teaches the
model nothing; requests the same reply already finished were recorded as each
one landed. `POST /v1/audio/speech` is one request, so an abandoned one
records nothing.

## Watching for changes

`/api/events` is a WebSocket that says which read went stale, so a page keeps
one socket open instead of polling. Each frame is `{"type": kind}` with
`kind` one of `models`, `voices`, `references` or `settings`; the client
re-reads that endpoint. A download in flight ticks `models` every half second
while its progress moves. A `ping` frame goes out after 25 s of silence. The
key travels as it does on the live socket, in the subprotocol list.

## Why a stream is MP3

`/api/speak/live` answers MP3: a bare sequence of self-describing frames with
no container, no length field and no index — the only honest thing to send
when the length is not known yet. A WAV stream has to declare a length before
the audio exists, and a general-purpose player given the maximal one waits for
a six-hour file; FLAC and OGG need a size or an index first, so they are not
offered. `bitrate` in the `ready` frame is the one measurement that exists
before the first sample, and is what turns a byte count into a duration.

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
| 503    | `BACKEND_MISSING`      | the model's backend needs an extra that is not installed; the message names it   |

## Settings over the API

`PUT /api/settings` takes any subset; a field that fails validation keeps its
previous value, and the reply names it under `ignored` beside `reloaded`,
which says whether resident models were dropped to adopt a thread count or
execution provider.

`text_rules` is what the four text switches default to when a request leaves
them out: a list of `{model, language, normalize_text, expand_numbers,
convert_script, taiwan_readings}` — `model` a catalog id or null for every
model, `language` a tag the request's resolved language must equal or extend
(`zh` covers `zh-TW`) or null, each switch `true`, `false` or null for "the
pipeline's call". Rules cascade per switch, the most specific one that says
something winning (model and language over either alone; equal ones, the
later). The list is replaced whole, and a rule naming a model the catalog
lacks refuses it whole under `ignored`.

## Versioning

`GET /health` reports `api_version`, bumped when a route, field or header a
client reads changes shape; the app's release `version` beside it says nothing
about the wire. `api_version` is 5. The integration refuses to set up on a
mismatch, so the app and the integration are released as a pair: the app
first, then the integration that requires the new number. A field a client did
not know about is not a change of shape, so adding one does not bump it.

## For integration authors

On Home Assistant OS the app announces itself through Supervisor discovery
with `{host, port, api_key}` under the service `cortex_tts`; the key is
generated on first start. Whenever the set of voices changes — a model
downloaded or deleted, a reference added or removed — the app fires
`cortex_tts_models_changed` on the Home Assistant event bus, which is how the
integration adds and removes entities without a reload.
