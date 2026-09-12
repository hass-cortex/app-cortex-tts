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

  for (const id of ["norm", "conv"]) {
    $(id).addEventListener("click", () => {
      $(id).setAttribute("aria-pressed", String(!pressed($(id))));
      preview.syncHint();
      preview.refresh();
    });
  }

  // The settings panel offers the same models and voices the rest of the page
  // does, including the ones a download finishing has just added.
  models.whenCatalogChanges(() =>
    settings.syncChoices(models.catalog(), models.knownVoices()));

  // Changing the default model or voice moves where the composer's pickers
  // open, which is the same fact read through a different endpoint.
  settings.whenSaved(() => models.loadDefaults());

  // The voice is the only place a language is declared, so the picker is what
  // moves the passes.
  models.whenVoiceChanges(() => preview.syncPassesToVoice(models.selectedVoiceLanguage()));
}

async function load() {
  try {
    const health = await (await fetch(url("/health"))).json();
    $("version").textContent = `v${health.version}`;
    settings.showProviders(health);
  } catch { $("version").textContent = "offline"; }

  try {
    await settings.load();
    await models.loadDefaults();
    await models.refreshModels();
    await models.refreshVoices();
    await clones.refresh();
    await preview.refresh();
  } catch (err) {
    msg($("speakMsg"), err.message, "err");
  }
}

wire();
preview.syncHint();
load();
