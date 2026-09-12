# A streamed format must not have to declare a length

`/api/speak/stream` opened with a WAV header, because WAV is what the rest of
the app produces and the header could be written before the audio existed. It
could — by declaring the maximal length, `0xFFFFFFFF` in both size fields,
which every reference on streaming WAV suggests and which `ffprobe` reads
correctly.

Real players are not ffprobe.

## What it looked like

Reported from a Windows desktop player (HASS.Agent): a reply played its first
sentence, stalled for the whole rest of the reply, then played everything at
once when the stream ended. Switching that model to `buffered` — a complete
file, real sizes, a Content-Length — played it smoothly, which is what makes
the header rather than the timing the cause.

Home Assistant serves a streamed TTS with `web.StreamResponse` and no
Content-Length, so the response is chunked. The player therefore receives a
file with no length that claims, in its own header, to be about four
gigabytes and six hours long. A player that sizes its pre-roll against the
declared duration waits for a fraction of six hours; one that waits for a seek
table waits for the end. Both produce exactly what was reported.

The ESP32 speaker on the same system was unaffected, because its firmware is
written for Home Assistant's streaming TTS and does not believe the header.
One consumer coping is not the same as the format being right.

## The decision

A format belongs on the streaming endpoint only if it can be written without
knowing how long the audio will be. MP3 is a bare sequence of frames, each
carrying its own header; there is no container, no length field and no index.
It is what every internet radio stream has used for thirty years, and it is
now this endpoint's default.

WAV stays available for a caller that wants raw PCM and knows what it is
asking for. FLAC and OGG are refused with a message that names what does
work: both need a size or a seek table in a header that would have to be
written first.

`X-Cortex-Bitrate` goes out with the response, because it is the one
measurement that exists before the first sample. At a constant bitrate a
consumer converts a byte count into a duration without decoding anything,
which is what the integration's buffering arithmetic needs and what a WAV
stream gave it for free.

## What it costs

192 kbps mono, measured at 0.23% of one core for ten seconds of speech — a
rounding error next to the model's two saturated threads, on a host where CPU
contention was already shown to matter. A quarter of raw PCM over the wire.

It is lossy, and the audio is usually transcoded once more before it reaches a
speaker, so a lossy step now feeds a second one. 192 kbps for mono speech is
well past transparent and the margin is deliberate; 128 would have been enough
and was not worth the argument.

One encoder per request means each request's audio begins with LAME's own
encoder delay, so the boundaries between requests gain a few tens of
milliseconds of silence. Those boundaries fall between sentences — the batch
that `MAX_REQUEST_CHARS` cuts is whole sentences — where a short pause is what
speech does anyway.

## What was rejected

**Declaring a different length.** Zero is read by common Windows decoders as
an empty file, which is worse than a long one. A plausible finite length —
ten minutes, say — risks truncating a reply at the declared size, and a
truncated answer is worse than a stuttering one.

**Leaving it to the consumer.** Home Assistant will transcode a TTS stream for
a player that declares what it supports; this player declares nothing. Waiting
for every player to improve is not a fix.
