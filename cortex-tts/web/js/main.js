// Composition root: it owns the wiring between panels, and nothing else.

import { url } from "./api.js";
import { $, msg, pressed } from "./dom.js";
import * as clones from "./clones.js";
import * as models from "./models.js";
import * as preview from "./preview.js";
import * as samples from "./samples.js";
import * as settings from "./settings.js";
import * as speak from "./speak.js";
import * as theme from "./theme.js";

function wire() {
  theme.init();
  models.init();
  clones.init();
  speak.init();
  settings.init();

  const clearSamples = samples.init(preview.refresh);
  $("text").addEventListener("input", () => {
    clearSamples();
    preview.schedule();
  });

  // Number expansion and bare numbers are the reader's own switches; the two Chinese rewrites
  // are the language's, shown only when the text is Chinese, and a click on
  // one overrides what the language decided.
  for (const id of ["norm", "num"]) {
    $(id).addEventListener("click", () => {
      $(id).setAttribute("aria-pressed", String(!pressed($(id))));
      preview.refresh();
    });
  }
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
      .catch((err) => msg($("modelMsg"), err.message, "err")));

  // The voice's own language is what the text is read in when the language
  // field is empty, so the column follows the voice. Whether there is a
  // voice at all is what makes Speak available.
  models.whenVoiceChanges(() => {
    speak.syncButton();
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
    ["speakMsg", preview.refresh()],
  ];
  const results = await Promise.allSettled(sections.map(([, task]) => task));
  results.forEach((result, i) => {
    if (result.status === "rejected") msg($(sections[i][0]), result.reason.message, "err");
  });
}

wire();
load();
