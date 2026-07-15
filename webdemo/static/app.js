(() => {
  const $ = (id) => document.getElementById(id);

  const runBtn = $("run-btn");
  const windowCountEl = $("window-count");
  const mockBanner = $("mock-banner");
  const errorBanner = $("error-banner");
  const resultsEl = $("results");

  const cpuFill = $("cpu-fill");
  const cpuCount = $("cpu-count");
  const cpuTime = $("cpu-time");
  const npuFill = $("npu-fill");
  const npuCount = $("npu-count");
  const npuTime = $("npu-time");
  const npuStageLabel = $("npu-stage-label");

  let pollTimer = null;
  let knownLimit = null;

  const stageName = (stage) => {
    if (stage === "encoder") return "Reading traffic patterns (encoder pass)";
    if (stage === "decoder") return "Reconstructing & scoring (decoder pass)";
    return "";
  };

  const fmtSeconds = (ms) => (ms / 1000).toFixed(2) + " s";

  function render(status) {
    mockBanner.hidden = !status.mock;

    if (status.config && status.config.limit && status.config.limit !== knownLimit) {
      knownLimit = status.config.limit;
      windowCountEl.textContent = knownLimit;
    }

    const cpu = status.cpu || {};
    const npu = status.npu || {};

    const cpuTotal = cpu.total || knownLimit || 1;
    const cpuPct = Math.min(100, (100 * (cpu.done || 0)) / cpuTotal);
    cpuFill.style.width = cpuPct + "%";
    cpuCount.textContent = `${cpu.done || 0} / ${cpuTotal}`;
    cpuTime.textContent = fmtSeconds(cpu.elapsed_ms || 0);

    // NPU runs two sequential passes (encoder, then decoder) over the same
    // N windows; treat that as one combined 0-100% bar so it's easy to read.
    const npuTotal = npu.total || knownLimit || 1;
    const stageOffset = npu.encoder_done ? npuTotal : 0;
    const npuDoneCombined = stageOffset + (npu.done || 0);
    const npuPct = Math.min(100, (100 * npuDoneCombined) / (npuTotal * 2));
    npuFill.style.width = npuPct + "%";
    npuCount.textContent = `${npu.done || 0} / ${npuTotal}`;
    npuTime.textContent = fmtSeconds(npu.elapsed_ms || 0);
    npuStageLabel.textContent = status.state === "running" ? stageName(npu.stage) : "";

    if (status.state === "error") {
      errorBanner.hidden = false;
      errorBanner.textContent = "Something went wrong: " + status.error;
      resultsEl.hidden = true;
    } else {
      errorBanner.hidden = true;
    }

    if (status.state === "done" && status.result) {
      resultsEl.hidden = false;
      renderResults(status.result);
    } else if (status.state !== "error") {
      resultsEl.hidden = true;
    }

    const running = status.state === "running";
    runBtn.disabled = running;
    runBtn.textContent = running ? "Running…" : "Run the demo";

    if (!running && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function renderResults(r) {
    $("speedup-value").textContent = r.speedup ? r.speedup.toFixed(1) : "—";
    $("cpu-auc").textContent = fmtMetric(r.cpu.auc);
    $("npu-auc").textContent = fmtMetric(r.npu.auc);
    $("cpu-f1").textContent = fmtMetric(r.cpu.f1);
    $("npu-f1").textContent = fmtMetric(r.npu.f1);
    $("corr-value").textContent = r.npu.corr_vs_pytorch != null
      ? (r.npu.corr_vs_pytorch * 100).toFixed(1) + "%"
      : "—";

    const body = $("examples-body");
    body.innerHTML = "";
    (r.examples || []).forEach((ex) => {
      const tr = document.createElement("tr");
      const badgeClass = ex.label === "attack" ? "badge-attack" : "badge-normal";
      tr.innerHTML = `
        <td>#${ex.window}</td>
        <td><span class="badge ${badgeClass}">${ex.label}</span></td>
        <td>${ex.cpu_score.toFixed(3)}</td>
        <td>${ex.npu_score.toFixed(3)}</td>
      `;
      body.appendChild(tr);
    });
  }

  const fmtMetric = (v) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(3));

  async function poll() {
    try {
      const res = await fetch("/api/status");
      const status = await res.json();
      render(status);
    } catch (e) {
      console.error("status poll failed", e);
    }
  }

  async function startRun() {
    runBtn.disabled = true;
    runBtn.textContent = "Starting…";
    resultsEl.hidden = true;
    errorBanner.hidden = true;
    try {
      const res = await fetch("/api/run", { method: "POST" });
      if (res.status === 409) {
        return; // a run is already in progress; polling will reflect it
      }
    } catch (e) {
      errorBanner.hidden = false;
      errorBanner.textContent = "Could not reach the demo server: " + e;
      runBtn.disabled = false;
      runBtn.textContent = "Run the demo";
      return;
    }
    if (!pollTimer) pollTimer = setInterval(poll, 300);
    poll();
  }

  runBtn.addEventListener("click", startRun);

  // Initial status fetch, and resume polling if a run is already underway
  // (e.g. the page was reloaded mid-run).
  poll().then(() => {
    fetch("/api/status").then((r) => r.json()).then((status) => {
      if (status.state === "running" && !pollTimer) {
        pollTimer = setInterval(poll, 300);
      }
    });
  });
})();
