# Summary_Gen


# Installation

### Clone the repo:

```bash
git clone https://github.com/jchiru21/Summary_Gen.git
cd Summary_Gen
```

### Create a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⚠️ If you have a GPU, install the matching CUDA-enabled PyTorch wheel from [PyTorch.org](https://pytorch.org/get-started/locally/).

---

# Running the App

Start the Flask server:

```bash
python app.py
```

Open your browser at:
[http://localhost:7860](http://localhost:7860)

Paste an article, optionally add a question, then click **Summarize + QA**.

---

# Project Structure

```
tripletgen-flask/
├─ app.py                  # Flask entrypoint
├─ requirements.txt        # Python dependencies
├─ tripletgen/
│  └─ summarize_and_qa.py  # Core summarizer + QA logic
├─ templates/
│  └─ index.html           # Frontend (Flask Jinja template)
├─ static/
│  └─ main.js              # Frontend JS
├─ checkpoints_clean/      # (Optional) local reranker checkpoints (ignored in Git)
└─ README.md               # You are here
```

---

# Models Used

* Generator: `facebook/bart-large-cnn`
* Reranker: `roberta-base` + fine-tuned checkpoints (optional)
* NLI: `facebook/bart-large-mnli`
* QA: `distilbert-base-cased-distilled-squad`

The first run will download these models from Hugging Face and cache them locally.

---

# Balanced vs Conservative

* **Balanced** → keeps more summaries, slightly looser filtering.
* **Conservative** → stricter entailment checks, fewer but safer summaries.

Both modes can be reproduced locally from the pipeline.

---

# Checkpoints Notice

This repo ignores large `.pt` files (`checkpoints_clean/`) using `.gitignore`.

* To use your fine-tuned reranker, place `reranker_ckpt_best_*.pt` in `./checkpoints_clean/`.
* Otherwise, the app falls back to the base `roberta-base`.


---

# Contributing

PRs welcome! Open an issue to discuss major changes.



