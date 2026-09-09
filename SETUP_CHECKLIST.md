# SETUP CHECKLIST — Phase 1 (Console Demo)

Print this page. Check off each item as you go.

---

## Before you start

- [ ] Python 3.11 or newer installed  
  Check: `python --version`

- [ ] You're in the project folder  
  Check: `dir` (Windows) or `ls` — you should see `agent/`, `demo/`, `README.md`

---

## 1. Get a free Groq API key

- [ ] Go to **console.groq.com** and sign up (free, no credit card)
- [ ] Click **API Keys** → **Create API Key**
- [ ] Copy the key — it starts with `gsk_`

---

## 2. Set up Python environment

```
python -m venv .venv
.venv\Scripts\activate          ← Windows
source .venv/bin/activate        ← Mac / Linux
pip install -e ".[dev]"
```

- [ ] Virtual environment created (`.venv/` folder appears)
- [ ] Dependencies installed (no red errors after `pip install`)

---

## 3. Create your .env file

```
copy .env.example .env          ← Windows
cp .env.example .env             ← Mac / Linux
```

- [ ] `.env` file created
- [ ] Opened `.env` in a text editor
- [ ] Replaced `gsk_REPLACE_WITH_YOUR_GROQ_KEY` with your real Groq key
- [ ] Saved the file

---

## 4. Run the tests (optional but recommended)

```
pytest tests/ -v -m "not live"
```

- [ ] All tests pass (no red FAILED lines)

---

## 5. Run the demo

```
python demo/console_demo.py
```

- [ ] Agent greets you and asks the first question
- [ ] You can type answers and it responds sensibly
- [ ] It asks a clarifying question when you give a vague answer (try: "sometime soon")
- [ ] It says goodbye when it has all the answers

---

## 6. Check what was saved

```
python scripts/view_db.py
```

- [ ] Your answers appear in the table

---

## ✅ Phase 1 complete!

**Next step:** tell your Antigravity agent to start **Phase 2** — adding real speech  
(mic input via Groq Whisper, spoken replies via Piper TTS).

You'll need to:
- Install Piper: `pip install piper-tts` then download a voice model
- No new API keys needed for Phase 2

---

## Phase 3 checklist (for later — real phone calls)

- [ ] Sign up at **twilio.com** (free trial, ~30 days)
- [ ] Verify your own phone number in the Twilio console (OTP)
- [ ] Copy Account SID, Auth Token, Phone Number into `.env`
- [ ] Install Cloudflare Tunnel: **developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads**
- [ ] Run: `cloudflared tunnel --url http://localhost:8000`
- [ ] Paste the tunnel URL into Twilio's webhook settings
