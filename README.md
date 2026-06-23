# AlignAgent 🎯

An **agentic RAG** system that analyses your resume against a job description and produces a full skill gap report — complete with a week-by-week learning plan, real learning resources, interview prep questions, and resume improvement suggestions.

Uses **Groq's free API** for the LLM and local FastEmbed for embeddings. Requires a free Groq API key.

---

## What it does

- Compares your resume against any job description
- Scores your readiness (0–100%)
- Builds a week-by-week study plan for every skill gap
- Finds free learning resources per skill
- Generates likely interview questions
- Suggests specific resume improvements
- Lets you compare your fit across multiple jobs side-by-side
- Tracks which skills you've learned over time

---

## How it works (Agentic RAG Pipeline)

```
Upload resume + JD
       ↓
Generate smart retrieval queries  (LLM writes its own search queries)
       ↓
Multi-query retrieval              (4 queries × 2 workspaces, deduplicated)
       ↓
Document grading                  (irrelevant chunks filtered out — CRAG)
       ↓
Web search fallback               (DuckDuckGo if local docs are sparse)
       ↓
Generate skill gap report         (structured JSON with plan + resources)
       ↓
Find learning resources           (web search per missing skill)
       ↓
Self-reflection                   (agent audits its own report, retries if needed)
       ↓
Save report
```

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Groq API — `llama-3.3-70b-versatile` (free tier) |
| Embeddings | FastEmbed — `BAAI/bge-small-en-v1.5` (384-dim, local) |
| Vector store | FAISS (local) |
| RAG framework | LangChain + LangGraph |
| Agentic graph | LangGraph (8 nodes, conditional edges, retry loops) |
| UI | Streamlit |
| Web search | DuckDuckGo (free) |

---

## Setup

### 1. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) and create an API key. The free tier is sufficient.

### 2. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/AlignAgent.git
cd AlignAgent
pip install -r requirements.txt
```

### 3. Set up environment

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 4. Ingest sample documents

```bash
python ingest_workspace.py --workspace candidate_demo --path sample_docs/resume.txt
python ingest_workspace.py --workspace internship_demo --path sample_docs/internship.txt
```

---

## Run

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### CLI agent (uses pre-ingested workspaces)

```bash
python run_agent.py
```

Saves report to `reports/final_gap_report.json`

### FastAPI server

```bash
uvicorn main:api_app --reload
```

Then `POST /analyze` with `{"candidate_id": "...", "internship_id": "..."}`.

---

## Project structure

```
AlignAgent/
├── app.py                  # Streamlit UI (all features)
├── graph.py                # LangGraph agentic pipeline (8 nodes)
├── pipeline.py             # Bridge: Streamlit upload → agentic graph
├── analyzer.py             # Quick 2-step LLM analyser
├── ingestion.py            # Multi-tenant FAISS ingestion pipeline
├── ingest_workspace.py     # CLI ingestion entry point
├── skill_gap_analyzer.py   # Core analysis logic + fallback
├── export.py               # HTML + JSON report export
├── progress.py             # Skill progress tracker (persistent)
├── main.py                 # FastAPI REST API
├── run_agent.py            # CLI agent runner
├── sample_docs/            # Example resume + internship description
└── requirements.txt
```

---

## UI Features

| Feature | Description |
|---|---|
| ⚡ Quick mode | 2-step LLM analysis (~40s) |
| 🧠 Deep mode | Full 8-node agentic graph with real-time progress (~2-3 min) |
| 🌐 URL scraping | Paste a LinkedIn/Indeed URL instead of text |
| 📅 Learning plan | Week-by-week plan with tasks and mini-projects |
| 📚 Resources | Free courses, docs, and videos per missing skill |
| 🎤 Interview prep | 6 likely interview questions based on the gap |
| 📝 Resume tips | 5 specific edits to strengthen your resume |
| ⚖️ Job comparison | Compare your fit across up to 3 jobs side-by-side |
| 📊 Progress tracker | Mark skills as learned — score updates automatically |
| 📥 Export | Download report as HTML (printable) or JSON |

---

## Agentic patterns implemented

| Pattern | Description |
|---|---|
| Dynamic query generation | LLM writes its own retrieval queries instead of hardcoded strings |
| Multi-query retrieval | Multiple targeted queries per workspace, results deduplicated |
| Corrective RAG (CRAG) | Each retrieved chunk is scored for relevance; low scores filtered out |
| Web search fallback | DuckDuckGo fires automatically if local docs are too sparse |
| Self-reflection loop | Agent audits its own report for hallucinations; retries retrieval if needed |

---

## License

MIT
