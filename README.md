# rag-assistant

Local-first personal RAG assistant.

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (dim=384)
- **Vector store**: `turbovec` (`TurboQuantIndex`, 4-bit quantization)
- **LLM**: DeepSeek (OpenAI-compatible chat API)
- **Backend**: FastAPI
- **Frontend**: single-file `index.html`, dark theme, no frameworks
- **Storage**: local `data/index.tq` + `metadata.json` + `memory.json`

Everything except the LLM call runs locally.

## Install

```bash
cd /home/ishu/Desktop/rag-assistant
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...    # required for /chat
```

## Ingest documents

Drop PDFs / `.txt` / `.md` / `.docx` into `data/raw/`, then:

```bash
cd backend
python ingest.py --source ../data/raw --index ../data/index.tq
```

The pipeline is incremental — files already in the index (by path + mtime + size signature)
are skipped on subsequent runs.

## Run the server

```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>.

## Intelligence layer

Every `/chat` call runs through:

1. **Intent router** (`intent.py`) → one of `knowledge | task | calendar | meta`.
   - `knowledge` uses RAG retrieval.
   - `task` (write email, summarise, translate) bypasses strict RAG grounding.
   - `calendar` returns an `.ics` block extracted from natural language.
   - `meta` answers questions about the assistant itself.
2. **Memory layer** (`memory.py`) — rolling `memory.json` with `facts` + last 20 `recent_queries`.
   Top-3 relevant facts (cosine similarity over embeddings) are injected into every prompt.
   To persist a new fact, send `remember` in the chat body:
   ```json
   {"query": "noted", "remember": "User prefers DeepSeek over GPT-4"}
   ```
3. **Structured output mode** — if the query contains `give me a table`, `compare`, or
   `list with details`, DeepSeek is called in JSON mode and the frontend renders
   the response as an HTML table.
4. **Citation fidelity check** — every knowledge-intent answer is scanned sentence
   by sentence; sentences whose content tokens don't appear in the retrieved chunks
   are flagged with `⚠️` in the UI.

## API

- `GET  /status` → `{total_chunks, index_size_mb, documents_indexed, memory_facts, recent_queries}`
- `POST /ingest` → multipart `file=@doc.pdf`
- `POST /chat`   → `{query, conversation_history?, remember?}`

## Layout

```
rag-assistant/
  backend/
    config.py        # constants
    ingest.py        # CLI + library ingestion
    retriever.py     # TurboVec + sentence-transformers
    intent.py        # intent router + structured-output detector
    memory.py        # rolling memory + relevance ranking
    llm.py           # DeepSeek client + prompts + citation check
    main.py          # FastAPI app
  frontend/
    index.html       # dark chat UI
  data/
    raw/             # source documents go here
    index.tq         # TurboVec index (built on first ingest)
    metadata.json    # chunk → source mapping
    memory.json      # facts + recent_queries
  requirements.txt
```
