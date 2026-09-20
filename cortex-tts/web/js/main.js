// Composition root: it owns the wiring between panels, and nothing else.

import { url } from "./api.js";
import { $, msg, pressed, soloAudio } from "./dom.js";
import * as clones from "./clones.js";
import * as live from "./live.js";
import * as models from "./models.js";
import * as updates from "./updates.js";
import * as preview from "./preview.js";
import * as samples from "./samples.js";
import * as settings from "./settings.js";
import * as theme from "./theme.js";

function wire() {
  // Whichever player starts takes the room. One listener rather than a call
  // in each panel: a rule about the page belongs to the page, and a panel
  // that forgets to make the call is a bug nothing catches.
  document.addEventListener(
    "play",
    (e) => {
      if (e.target instanceof HTMLAudioElement) soloAudio(e.target);
    },
    true,
  );

  theme.init();
  models.init();
  clones.init();
  live.init();
  settings.init();

  const clearSamples = samples.init(preview.refresh);
  $("text").addEventListener("input", () => {
    clearSamples();
    preview.schedule();
  });

  // Number expansion is the reader's own switch. Bare numbers are the
  // model's decision and the two Chinese rewrites the language's, shown
  // only when the text is Chinese; a click on any of those overrides it.
  $("norm").addEventListener("click", () => {
    $("norm").setAttribute("aria-pressed", String(!pressed($("norm"))));
    preview.refresh();
  });
  $("num").addEventListener("click", () => preview.toggle("expand_numbers"));
  $("conv").addEventListener("click", () => preview.toggle("convert_script"));
  $("tw").addEventListener("click", () => preview.toggle("taiwan_readings"));

  // The settings panel offers the same models and voices the rest of the page
  // does, including the ones a download finishing has just added.
  models.whenCatalogChanges(() =>
    settings.syncChoices(models.catalog(), models.knownVoices()));

  // Changing the default model or voice moves where the composer's pickers
  // open, and a smaller resident bound or a rebound session evicts what is
  // loaded — the same facts read through different endpoints.
  settings.whenSaved(() =>
    models.loadDefaults()
      .then(() => models.refreshModels())
      .then(() => preview.refresh())
      .catch((err) => msg($("modelMsg"), err.message, "err")));

  // One socket says which section went stale; nothing polls.
  updates.init();

  // The voice's own language is what the text is read in when the language
  // field is empty, so the column follows the voice. Whether there is a
  // voice at all is what makes speaking available.
  models.whenVoiceChanges(() => {
    live.syncButton();
    preview.refresh();
  });
  // The upload form's language list is the models' business, not its own —
  // and its value follows the Language filter until someone sets it there.
  models.whenModelsChange(clones.syncLanguages);
  $("language").addEventListener("change", () => {
    clones.syncLanguages();
    preview.refresh();
  });
}

async function load() {
  try {
    const health = await (await fetch(url("/health"))).json();
    $("version").textContent = `v${health.version}`;
    settings.showProviders(Object.values(health.providers_in_use || {}));
  } catch { $("version").textContent = "offline"; }

  // The pickers prefer the configured defaults, so those are read before the
  // lists they apply to. Every other section stands on its own: one failing
  // must not blank the rest, and each says so in its own slot.
  const sections = [
    ["setMsg", settings.load()],
    ["modelMsg", models.loadDefaults()
      .then(() => models.refreshModels())
      .then(() => models.refreshVoices())],
    ["refMsg", clones.refresh()],
    ["playerMsg", preview.refresh()],
  ];
  const results = await Promise.allSettled(sections.map(([, task]) => task));
  results.forEach((result, i) => {
    if (result.status === "rejected") msg($(sections[i][0]), result.reason.message, "err");
  });
}

wire();
load();
