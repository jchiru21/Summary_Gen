document.addEventListener("DOMContentLoaded", () => {
  const articleEl = document.getElementById("article");
  const questionEl = document.getElementById("question");
  const slider = document.getElementById("summarySlider");
  const tokenCount = document.getElementById("tokenCount");
  const rightTokenCount = document.getElementById("rightTokenCount");
  const sentCount = document.getElementById("sentCount");
  const wordCount = document.getElementById("wordCount");
  const summarizeBtn = document.getElementById("summarize");
  const summarizeSpinner = document.getElementById("summarizeSpinner");
  const resultArea = document.getElementById("resultArea");
  const statusEl = document.getElementById("status");
  const clearBtn = document.getElementById("clear");
  const pasteBtn = document.getElementById("pasteBtn");
  const fileInput = document.getElementById("fileInput");
  const presetMarkers = document.querySelectorAll(".preset-marker");

  const PRESETS = [
    { name: "Concise", val: 20, range: [0, 35] },
    { name: "Medium", val: 55, range: [36, 74] },
    { name: "Detailed", val: 90, range: [75, 100] }
  ];

  function sliderToParams(val) {
    const maxTokens = Math.round(96 + (val / 100) * (512 - 96));
    const lengthPenalty = 1.0 + (val / 100) * 0.8;   // 1.0 -> 1.8
    const numBeams = Math.round(2 + (val / 100) * 8); // 2..10
    const numReturn = Math.round(1 + (val / 100) * 7); // 1..8 candidates
    const doSample = val > 80 ? true : false; // sample only for very long
    return {
      max_new_tokens: maxTokens,
      length_penalty: Number(lengthPenalty.toFixed(2)),
      num_beams: Math.max(2, numBeams),
      do_sample: doSample,
      num_return_sequences: Math.max(1, numReturn)
    };
  }

  function updateMeta() {
    const text = articleEl.value || "";
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const sents = text.trim() ? text.trim().split(/(?<=[.!?])\s+/).length : 0;
    wordCount.textContent = `${words} words`;
    sentCount.textContent = `${sents} sentences`;
    const val = parseInt(slider.value, 10);
    const params = sliderToParams(val);
    tokenCount.textContent = `~${params.max_new_tokens} tokens`;
    if (rightTokenCount) rightTokenCount.textContent = `~${params.max_new_tokens} tokens`;
    updateActivePreset(val);
  }

  function updateActivePreset(val) {
    let activeIndex = 1;
    for (let i = 0; i < PRESETS.length; i++) {
      const p = PRESETS[i];
      if (val >= p.range[0] && val <= p.range[1]) {
        activeIndex = i;
        break;
      }
    }
    presetMarkers.forEach((el, idx) => {
      if (idx === activeIndex) {
        el.classList.add("active");
        el.setAttribute("aria-selected", "true");
      } else {
        el.classList.remove("active");
        el.setAttribute("aria-selected", "false");
      }
    });
  }

  // initialize
  slider.addEventListener("input", updateMeta);
  articleEl.addEventListener("input", updateMeta);
  updateMeta();

  // presets click behavior
  presetMarkers.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const v = Number(btn.getAttribute("data-val"));
      slider.value = v;
      updateMeta();
      slider.focus();
    });
    btn.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        btn.click();
      }
    });
  });

  // paste handler
  pasteBtn.addEventListener("click", async () => {
    try {
      const text = await navigator.clipboard.readText();
      articleEl.value = text;
      updateMeta();
    } catch (e) {
      console.warn("Clipboard read failed", e);
      alert("Paste failed — allow clipboard permissions or paste manually.");
    }
  });

  // file upload
  fileInput.addEventListener("change", (ev) => {
    const f = ev.target.files && ev.target.files[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = () => { articleEl.value = r.result; updateMeta(); };
    r.readAsText(f);
  });

  // clear
  clearBtn.addEventListener("click", () => {
    articleEl.value = "";
    questionEl.value = "";
    updateMeta();
    resultArea.innerHTML = `<div class="placeholder">No result yet. Click <strong>Summarize</strong>.</div>`;
    statusEl.textContent = "Idle";
  });

  // summarize click handler
  summarizeBtn.addEventListener("click", async () => {
    const article = articleEl.value.trim();
    if (!article) { alert("Please paste or upload an article first."); return; }
    const question = questionEl.value.trim() || null;
    const sliderVal = parseInt(slider.value, 10);
    const params = sliderToParams(sliderVal);

    summarizeSpinner.classList.remove("hidden");
    summarizeBtn.setAttribute("disabled", "true");
    statusEl.textContent = "Working...";

    try {
      const payload = {
        article,
        question,
        max_new_tokens: params.max_new_tokens,
        num_beams: params.num_beams,
        length_penalty: params.length_penalty,
        do_sample: params.do_sample,
        num_return_sequences: params.num_return_sequences,
        K: params.num_return_sequences
      };

      const resp = await fetch("/api/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Server error: ${resp.status} ${text}`);
      }

      const data = await resp.json();
      const top = data.generative_top3 && data.generative_top3.length ? data.generative_top3[0] : null;
      const html = [];
      if (top) {
        html.push(`<div class="summary-block"><h4>Top summary</h4><p>${escapeHtml(top.candidate)}</p>`);
        html.push(`<div class="meta muted">Entail prob (top1): ${data.generative_top1_entail ?? "N/A"} • Fallback: ${data.fallback_used ? "yes" : "no"}</div></div>`);
      } else {
        html.push('<div class="placeholder">No summary produced.</div>');
      }
      if (data.fallback && data.fallback.extractive_summary) {
        html.push(`<div class="fallback-block"><h4>Extractive fallback</h4><p>${escapeHtml(data.fallback.extractive_summary)}</p></div>`);
      }
      if (data.qa_answer) {
        html.push(`<div class="qa-block"><h4>QA</h4><pre>${escapeHtml(JSON.stringify(data.qa_answer, null, 2))}</pre></div>`);
      }
      resultArea.innerHTML = html.join("\n");
      statusEl.textContent = "Done";
    } catch (err) {
      console.error(err);
      resultArea.innerHTML = `<div class="error">Error: ${escapeHtml(err.message)}</div>`;
      statusEl.textContent = "Error";
    } finally {
      summarizeSpinner.classList.add("hidden");
      summarizeBtn.removeAttribute("disabled");
    }
  });

  function escapeHtml(str) {
    if (typeof str !== "string") return String(str);
    return str.replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
  }
});
