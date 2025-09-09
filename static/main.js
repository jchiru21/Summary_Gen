/* static/main.js - vanilla JS; small, dependency free */
async function postJSON(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(data),
  });
  return res.json();
}

function el(tag, attrs={}, children=[]) {
  const e = document.createElement(tag);
  for(const k in attrs) {
    if(k === "class") e.className = attrs[k];
    else if(k === "text") e.textContent = attrs[k];
    else e.setAttribute(k, attrs[k]);
  }
  children.forEach(c => e.appendChild(c));
  return e;
}

document.addEventListener("DOMContentLoaded", () => {
  const runBtn = document.getElementById("run");
  const clearBtn = document.getElementById("clear");
  const articleInp = document.getElementById("article");
  const questionInp = document.getElementById("question");
  const resBox = document.getElementById("res");
  const status = document.getElementById("status");
  const preset = document.getElementById("preset");

  clearBtn.addEventListener("click", () => {
    articleInp.value = "";
    questionInp.value = "";
    resBox.innerHTML = '<div class="placeholder">No result yet. Click <strong>Summarize</strong> to run inference.</div>';
    status.className = "status idle";
    status.textContent = "Idle";
  });

  runBtn.addEventListener("click", async () => {
    const article = articleInp.value.trim();
    const question = questionInp.value.trim();
    if(!article) {
      alert("Paste an article first.");
      return;
    }
    status.className = "status running";
    status.textContent = "Running…";

    // small UX: disable while running
    runBtn.disabled = true;
    runBtn.textContent = "Running…";

    try {
      // attach preset for future server-side behavior if needed
      const payload = { article, question, preset: preset.value };
      const out = await postJSON("/api/summarize", payload);

      // handle errors
      if(out && out.error) {
        resBox.innerHTML = `<div class="placeholder">Error: ${out.error}</div>`;
      } else {
        // pretty render top3
        resBox.innerHTML = "";
        const top3 = (out.generative_top3 || []).slice(0,3);
        if(top3.length === 0) {
          resBox.innerHTML = '<div class="placeholder">No candidates returned.</div>';
        } else {
          top3.forEach((c,i) => {
            const card = el("div", {class:"candidate"});
            const title = el("div", {class:"meta"});
            title.appendChild(el("div", {text:`#${i+1} • combined=${(c.combined||0).toFixed(3)}  rerank=${(c.rerank_score||0).toFixed(3)}`}));
            card.appendChild(title);
            card.appendChild(el("h3", {text: `Candidate ${i+1}`}));
            card.appendChild(el("pre", {text: c.candidate || ""}));
            const controls = el("div", {class:"controls"});
            const copyBtn = el("button", {class:"small-btn", text:"Copy"});
            copyBtn.onclick = () => { navigator.clipboard.writeText(c.candidate||""); copyBtn.textContent = "Copied"; setTimeout(()=> copyBtn.textContent="Copy",1200); };
            const downloadBtn = el("button", {class:"small-btn", text:"Download"});
            downloadBtn.onclick = () => {
              const blob = new Blob([c.candidate||""], {type:"text/plain;charset=utf-8"});
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url; a.download = `summary_${i+1}.txt`; document.body.appendChild(a); a.click(); a.remove();
              URL.revokeObjectURL(url);
            };
            controls.appendChild(copyBtn);
            controls.appendChild(downloadBtn);
            card.appendChild(controls);
            resBox.appendChild(card);
          });
        }

        // render general info like entail / fallback / qa
        const metaWrap = el("div", {class:"card meta-wrap"});
        const general = el("div", {}, []);
        general.appendChild(el("div", {text: `Entail prob (top1): ${out.generative_top1_entail ?? "n/a"}`}));
        general.appendChild(el("div", {text: `Fallback used: ${out.fallback_used ? "yes": "no"}`}));
        if(out.fallback) {
          general.appendChild(el("div", {text: `Fallback extractive (snippet): ${out.fallback.extractive_summary?.slice(0,180) ?? ""}`}));
        }
        if(out.qa_answer) {
          general.appendChild(el("div", {text: `QA: ${out.qa_answer.answer ?? JSON.stringify(out.qa_answer)}`}));
        }
        resBox.appendChild(general);
      }
    } catch (err) {
      resBox.innerHTML = `<div class="placeholder">Network or server error: ${err.message}</div>`;
    } finally {
      status.className = "status idle";
      status.textContent = "Idle";
      runBtn.disabled = false;
      runBtn.textContent = "Summarize";
    }
  });
});
