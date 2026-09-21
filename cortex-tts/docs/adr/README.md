# Architecture decisions

One file per decision: the context, what was decided, the measurement that
justifies it, and what follows. `AGENTS.md` states each rule in a line and
points here for the reasoning; a rule's number changes here first.

| ADR                                                 | Decision                                                                                                       |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| [0001](0001-rtf-threshold-pacing.md)                | A live reply is paced by one measured real-time factor and one threshold                                       |
| [0002](0002-text-pipeline-keyed-by-language.md)     | The text pipeline is keyed by language; a pass that can misjudge is opt-in                                     |
| [0003](0003-overrun-trimming-and-stopping.md)       | An autoregressive model is stopped by punctuation, retried when it stops early, and trimmed only on short text |
| [0004](0004-references-are-audio-and-transcript.md) | A reference is audio and transcript, validated on the way in                                                   |
| [0005](0005-device-memory-and-providers.md)         | The card is read, never assumed: providers, residency, running out of memory                                   |
| [0006](0006-live-socket-transport.md)               | The live socket: MP3 without a length, bounded frames, legible delivery                                        |
| [0007](0007-measurements-belong-to-a-host.md)       | A real-time factor belongs to a host, not to a model                                                           |
