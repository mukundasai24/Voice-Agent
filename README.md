# Phone-Calling AI Voice Agent

An AI voice agent that places or receives real phone calls, asks a script of questions, understands varied answers, clarifies when needed, and saves the results as structured data.

Built with: **Pipecat** · **Twilio** · **Groq** (Whisper + Llama) · **Piper TTS** · **FastAPI** · **SQLite → Supabase**

> 🟢 **Phase 1 complete** — the conversation brain works in text-only console mode.
> Audio (Phase 2) and real phone calls (Phase 3) will be added next.

---

## What each folder does

```
voice-agent/
├── agent/          ← The brain: questions, state, extraction, storage
├── demo/           ← Runnable demos (Phase 1: console, Phase 2: mic)
├── interfaces/     ← Swappable components: STT, TTS, LLM, telephony (Phase 2+)
├── pipeline/       ← Pipecat pipeline + FastAPI server (Phase 3+)
├── scripts/        ← Utilities: view the database, initialize schema
├── tests/          ← Unit + integration tests (no API key needed for most)
└── tests/fixtures/ ← Sample transcript + expected JSON output
```

---

## Phase 1: Console Demo (text only — no audio, no phone)

This is where you start. It proves the entire conversation logic works before touching any telephony.

### Step 1 — Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com) → Sign up (free, no credit card)
2. Click **API Keys** → **Create API Key**
3. Copy the key (starts with `gsk_…`)

### Step 2 — Set up the project

```bash
# From the project root folder:

# Create a virtual environment (keeps your system Python clean)
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
# If that fails, try: pip install groq python-dotenv "pydantic>=2" pydantic-settings rich aiosqlite pytest pytest-asyncio
```

### Step 3 — Create your .env file

```bash
# Copy the template
copy .env.example .env      # Windows
# or: cp .env.example .env  # Mac/Linux
```

Open `.env` in any text editor and replace `gsk_REPLACE_WITH_YOUR_GROQ_KEY` with your real key. **That's the only thing you need to change for Phase 1.**

### Step 4 — Run the demo

```bash
python demo/console_demo.py
```

You'll see something like:

```
╭─────────────────────────────────────────╮
│  Phone-Calling AI Voice Agent           │
│  Phase 1 — Console Demo (text only)     │
╰─────────────────────────────────────────╯

🤖 Agent: Hello! I'm an automated booking assistant...
           Could I start by getting your full name, please?

You: ▌
```

Type your answers. The agent will ask clarifying questions if your answer is vague. When it has everything it needs, it says goodbye. Your answers are saved automatically.

### Step 5 — Check what was saved

```bash
python scripts/view_db.py
```

---

## Running the tests

```bash
# Fast tests (no API key needed — all LLM calls are mocked):
pytest tests/ -v -m "not live"

# All tests including live Groq API calls (requires GROQ_API_KEY in .env):
pytest tests/ -v
```

---

## How the conversation works

```
Agent asks a question
       │
       ▼
Caller replies  ──► LLM extracts: { value, confidence, clarifying_question }
       │
       ├── confidence ≥ 0.75 ──► Accept answer, move to next question
       │
       ├── confidence < 0.75, attempts remaining ──► Ask the clarifying question
       │
       └── confidence < 0.75, max attempts reached ──► Flag for human follow-up, move on
```

Every turn, the LLM is asked to return a small JSON object — not a free-form answer. That's what lets it understand "yeah sometime next week probably" as a real (if fuzzy) answer and generate a targeted clarifying question, rather than failing to match a keyword.

---

## Changing what the agent asks

Open [`agent/questions.py`](agent/questions.py) — that's the only file to edit for a different use-case. Change `QUESTIONS` and `CALL_SCHEMA`. Nothing else needs to change.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `AuthenticationError: 401` | API key not loaded | Check `.env` exists and has the real key (not the placeholder) |
| `ModuleNotFoundError: groq` | Dependencies not installed | Run `pip install -e ".[dev]"` inside your activated `.venv` |
| `ModuleNotFoundError: pydantic_settings` | Missing package | `pip install pydantic-settings` |
| LLM returns empty replies | Sending wrong message format | Check `DEBUG_LLM=true` in `.env` to see the raw prompt/response |
| `OperationalError: unable to open database` | Wrong working directory | Run scripts from the project root folder, not from a subfolder |
| Agent keeps clarifying forever | MAX_CLARIFY_ATTEMPTS too high | It's set to 2 in `extractor.py` — it will move on after 2 failed attempts |
| Saved JSON has `__NEEDS_FOLLOWUP__` values | Agent gave up on a required field | Check the logs for what the caller said vs. what was extracted |

---

## What's coming (Phase 2+)

- **Phase 2** — Real speech: mic input via Groq Whisper, spoken replies via Piper TTS
- **Phase 3** — Real phone call via Twilio Media Streams + Cloudflare Tunnel
- **Phase 4** — Validated JSON extraction + Supabase (Postgres) option
- **Phase 5** — Structured logging, error handling, full test coverage, polished README

---

## Security notes

- **Never commit `.env`** — it's in `.gitignore`. Commit `.env.example` (which has no real values) instead.
- **Never hardcode API keys** in `.py` files or comments.
- The local database file (`agent_calls.db`) is also git-ignored — don't commit it.
- If this project goes beyond personal testing, re-read Section 9 (Security & Legal) in the research document before calling anyone outside your verified test numbers.

---

## Cost

| Component | Cost |
|---|---|
| Everything in Phase 1 & 2 | **$0** — only Groq is needed (free tier) |
| Phase 3 (real calls) | **$0** — Twilio free trial covers calls to verified numbers |
| Production hosting (Phase 5+) | ~$7/month for an always-on server |
