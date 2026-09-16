"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  let catalog = [],
    presets = [],
    selectedJob = "",
    pending = false,
    request = null,
    olderJobs = [],
    nextJobBefore = null;
  async function api(path, body) {
    const response = await fetch(
      `/api/v1/control/${path}`,
      body === undefined
        ? {}
        : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          },
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(
        result.error?.message || result.error || `HTTP ${response.status}`,
      );
    return result.data;
  }
  function message(value, error = false) {
    $("job-message").textContent = value;
    $("job-message").className = error ? "error" : "muted";
  }
  function download(value, name) {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function operation() {
    return catalog.find((c) => c.path.join(" ") === $("operation").value);
  }
  function argv() {
    const command = operation();
    if (!command) return [];
    const args = [...command.path];
    for (const field of command.arguments) {
      const input = $(`arg-${field.id}`);
      if (!input) continue;
      if (!field.takes_value) {
        if (input.checked) args.push(`--${field.long}`);
        continue;
      }
      if (!input.value.trim()) continue;
      const values = field.multiple
        ? input.value
            .split("\n")
            .map((s) => s.trim())
            .filter(Boolean)
        : [input.value];
      if (field.repeat && field.long) {
        for (const value of values) args.push(`--${field.long}`, value);
      } else {
        if (field.long) args.push(`--${field.long}`);
        args.push(...values);
      }
    }
    return args;
  }
  function preview() {
    $("command-preview").textContent = `glr ${argv()
      .map((a) => (/\s/.test(a) ? JSON.stringify(a) : a))
      .join(" ")}`;
    request = null;
  }
  function renderForm() {
    const command = operation();
    $("operation-fields").replaceChildren();
    if (!command) return;
    $("operation-description").textContent =
      command.description || `Run ${command.path.join(" ")}`;
    for (const field of command.arguments) {
      const label = document.createElement("label");
      label.textContent =
        field.id.replaceAll("_", " ") + (field.required ? " *" : "");
      let input;
      if (!field.takes_value) {
        input = document.createElement("input");
        input.type = "checkbox";
        label.className = "flag";
      } else if (field.choices.length) {
        input = document.createElement("select");
        for (const value of ["", ...field.choices]) {
          const option = document.createElement("option");
          option.value = value;
          option.textContent = value || "Default";
          input.append(option);
        }
      } else if (field.multiple) {
        input = document.createElement("textarea");
        input.placeholder = "One value per line";
      } else {
        input = document.createElement("input");
        input.type = "text";
      }
      input.id = `arg-${field.id}`;
      input.required = field.required;
      input.title = field.help || field.id;
      if (field.takes_value && field.defaults.length)
        input.value = field.defaults.join(field.multiple ? "\n" : "");
      input.oninput = preview;
      label.append(input);
      $("operation-fields").append(label);
    }
    const training = ["train", "goal", "task"].includes(command.path[0]);
    $("save-preset").disabled = !training;
    preview();
  }
  async function launch(payload) {
    if (pending) return;
    pending = true;
    const fingerprint = JSON.stringify(payload);
    if (!request || request.fingerprint !== fingerprint)
      request = { fingerprint, id: `request-${crypto.randomUUID()}` };
    const buttons = [...document.querySelectorAll("#controls button")];
    buttons.forEach((b) => (b.disabled = true));
    try {
      message("Starting operation…");
      const job = await api("jobs", { ...payload, request_id: request.id });
      selectedJob = job.id;
      window.dispatchEvent(
        new CustomEvent("glr-select-job", { detail: { jobId: job.id } }),
      );
      message(`${job.id} · ${job.status}`);
      request = null;
      await refreshJobs();
    } catch (error) {
      message(error.message, true);
    } finally {
      pending = false;
      buttons.forEach((b) => (b.disabled = false));
      $("save-preset").disabled = !["train", "goal", "task"].includes(
        operation()?.path[0],
      );
    }
  }
  async function refreshPresets() {
    presets = await api("presets");
    $("presets").replaceChildren();
    for (const preset of presets) {
      const card = document.createElement("article");
      card.className = "preset";
      const title = document.createElement("h3");
      title.textContent = preset.title;
      const description = document.createElement("p");
      description.className = "muted";
      description.textContent = preset.description;
      const actions = document.createElement("div");
      actions.className = "preset-actions";
      const start = document.createElement("button");
      start.className = "primary";
      start.textContent = "Start training";
      start.onclick = () => launch({ preset: preset.id });
      const exportButton = document.createElement("button");
      exportButton.textContent = "Export preset";
      exportButton.onclick = () =>
        download(
          {
            ...preset,
            id: preset.id === "train.default" ? "train.imported" : preset.id,
          },
          `${preset.id}.json`,
        );
      actions.append(start, exportButton);
      card.append(title, description, actions);
      $("presets").append(card);
    }
  }
  async function refreshJobs() {
    const result = await api("jobs");
    if (!olderJobs.length) nextJobBefore = result.next_before;
    result.jobs = [
      ...new Map([...result.jobs, ...olderJobs].map((j) => [j.id, j])).values(),
    ];
    $("older-jobs").hidden = !nextJobBefore;
    $("jobs").replaceChildren();
    for (const job of result.jobs) {
      const button = document.createElement("button");
      button.textContent = `${job.observed_status || job.status} · ${job.requested_argv.join(" ")}`;
      const detail = document.createElement("small");
      detail.textContent = `${new Date(job.created_at_ms).toLocaleString()} · exit ${job.exit_code ?? "—"}`;
      button.append(detail);
      button.onclick = () => {
        selectedJob = job.id;
        window.dispatchEvent(
          new CustomEvent("glr-select-job", { detail: { jobId: job.id } }),
        );
        refreshJobs().catch((e) => message(e.message, true));
      };
      $("jobs").append(button);
    }
    if (!selectedJob && result.jobs.length) selectedJob = result.jobs[0].id;
    if (selectedJob) {
      const job = result.jobs.find((j) => j.id === selectedJob);
      $("selected-job").textContent = job
        ? `${job.id} · ${job.observed_status || job.status}`
        : selectedJob;
      const log = await api(
        `job-log?id=${encodeURIComponent(selectedJob)}&stream=${$("job-stream").value}`,
      );
      $("job-output").textContent = log.text || "No output yet.";
    }
  }
  $("operation").onchange = renderForm;
  $("operation-form").onsubmit = (e) => {
    e.preventDefault();
    launch({ argv: argv() });
  };
  $("save-preset").onclick = async () => {
    try {
      await api("presets", {
        schema_version: "glr.training-preset.v1",
        id: $("preset-id").value,
        title: $("preset-title").value,
        description: $("preset-description").value,
        argv: argv(),
      });
      await refreshPresets();
      message("Training preset saved.");
    } catch (e) {
      message(e.message, true);
      $("preset-id").closest("details").open = true;
    }
  };
  $("preset-import").onchange = async () => {
    try {
      const file = $("preset-import").files[0];
      if (!file) return;
      if (file.size > 65536) throw new Error("Preset exceeds 64 KiB.");
      await api("presets", JSON.parse(await file.text()));
      await refreshPresets();
      message("Preset imported; review it before starting training.");
    } catch (e) {
      message(e.message, true);
    } finally {
      $("preset-import").value = "";
    }
  };
  $("job-stream").onchange = () =>
    refreshJobs().catch((e) => message(e.message, true));
  document.querySelectorAll("[data-command]").forEach(
    (b) =>
      (b.onclick = () => {
        $("operation").value = b.dataset.command;
        renderForm();
        $("operation").focus();
      }),
  );
  $("older-jobs").onclick = async () => {
    const button = $("older-jobs");
    button.disabled = true;
    try {
      const page = await api(
        `jobs?before=${encodeURIComponent(nextJobBefore)}`,
      );
      olderJobs.push(...page.jobs);
      nextJobBefore = page.next_before;
      await refreshJobs();
    } catch (e) {
      message(e.message, true);
    } finally {
      button.disabled = false;
    }
  };
  async function tick() {
    try {
      if (!pending) await refreshJobs();
    } catch (e) {
      message(`Job history unavailable: ${e.message}`, true);
    } finally {
      setTimeout(tick, 1500);
    }
  }
  async function init() {
    try {
      const health = await fetch("/api/v1/health").then((r) => r.json());
      if (health.read_only) return;
      const data = await api("catalog");
      catalog = data.commands;
      for (const command of catalog) {
        const option = document.createElement("option");
        option.value = command.path.join(" ");
        option.textContent = command.path.join(" › ");
        $("operation").append(option);
      }
      $("operation").value = "train";
      renderForm();
      await refreshPresets();
      $("controls").hidden = false;
      tick();
    } catch (e) {
      message(e.message, true);
      setTimeout(init, 3000);
    }
  }
  init();
})();
