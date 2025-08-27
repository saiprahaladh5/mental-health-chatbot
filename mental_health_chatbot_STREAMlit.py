import os, time, random, html
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import expit
from together import Together

# =================== THEME ===================
st.set_page_config(page_title="Journaling Coach", layout="centered")
st.markdown("""
<style>
/* Color system */
:root{
  --bg:#0b1220;
  --panel:#0f172a;
  --border:#1f2937;
  --text:#e5e7eb;
  --muted:#94a3b8;
  --accent:#22c55e;
  --bubble-in:#111827;      /* assistant */
  --bubble-out:#2563eb;     /* user */
  --bubble-out-border:#1e40af;
}

html, body .main { background: var(--bg) !important; color: var(--text) !important; }
.block-container { padding-top: 10px; max-width: 860px; }

/* App bar */
.appbar {
  background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
  color:#062332; border-radius:18px; padding:12px 16px; font-weight:700;
  box-shadow: 0 8px 30px rgba(0,0,0,.25); margin-bottom:12px;
}

/* Emotion chips (subtle) */
.chips { display:flex; gap:.4rem; flex-wrap:wrap; margin:6px 0 12px }
.chip { padding:.34rem .6rem; border-radius:999px; border:1px solid var(--border); color:#cbd5e1;
        background:#0b1325; font-size:.9rem; }

/* Chat area */
.chatwrap {
  background: var(--panel); border:1px solid var(--border); border-radius:18px;
  padding: 10px 12px; min-height: 460px;
  box-shadow: 0 8px 40px rgba(0,0,0,.25);
}
.msg { display:flex; margin:12px 6px; }
.bubble {
  border-radius:16px; padding:12px 14px; line-height:1.5; max-width:80%;
  border:1px solid var(--border);
}
.assistant .bubble { background: var(--bubble-in); color: var(--text); }
.user      { justify-content:flex-end; }
.user .bubble {
  background: var(--bubble-out); color:#fff; border-color: var(--bubble-out-border);
  box-shadow: 0 4px 18px rgba(37,99,235,.35);
}
.meta { font-size:.78rem; color:var(--muted); margin-top:6px }
.footer { color:var(--muted); margin-top:10px; }
hr { border:none; height:1px; background:var(--border); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="appbar">🧠 Journaling Coach — WhatsApp-style, warm & practical</div>', unsafe_allow_html=True)

# =================== EMOTION MODEL ===================
EMOJI = {"joy":"😊","sadness":"😢","anger":"😠","fear":"😟","disgust":"🤢","surprise":"😮","love":"💖","optimism":"🌤️","neutral":"😐"}

@st.cache_resource(show_spinner=False)
def _load_emotion_model():
    m = "SamLowe/roberta-base-go_emotions"
    tok = AutoTokenizer.from_pretrained(m)
    mdl = AutoModelForSequenceClassification.from_pretrained(m)
    id2label = {int(k): v for k, v in mdl.config.id2label.items()} if isinstance(mdl.config.id2label, dict) else mdl.config.id2label
    groups = {
        "joy":{"amusement","excitement","joy","pride","relief","gratitude"},
        "sadness":{"sadness","grief","disappointment","remorse"},
        "anger":{"anger","annoyance","disapproval"},
        "fear":{"fear","nervousness","embarrassment","confusion"},
        "disgust":{"disgust"},
        "surprise":{"surprise","realization"},
        "love":{"love","caring","admiration","approval","desire"},
        "optimism":{"optimism","curiosity"},
        "neutral":{"neutral"},
    }
    idx = {g: [] for g in groups}
    for i in range(mdl.config.num_labels):
        lab = id2label[i].lower(); placed=False
        for g,fine in groups.items():
            if lab in fine: idx[g].append(i); placed=True; break
        if not placed: idx["neutral"].append(i)
    return tok, mdl, idx

try:
    TOKENIZER, EMO_MODEL, IDX = _load_emotion_model(); MODEL_OK=True
except Exception:
    TOKENIZER=EMO_MODEL=IDX=None; MODEL_OK=False

def detect_emotion_9(text:str):
    if not MODEL_OK: return "neutral",0.5,{"neutral":1.0}
    x = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad(): logits = EMO_MODEL(**x).logits
    p = expit(logits.detach().cpu().numpy()[0])
    scores = {g: float(sum(p[i] for i in idxs)) for g,idxs in IDX.items()}
    total = sum(scores.values()) or 1.0
    scores = {k:v/total for k,v in scores.items()}
    top = max(scores, key=scores.get)
    return top, scores[top], scores

# =================== TOGETHER CLIENT ===================
API_KEY = os.getenv("TOGETHER_API_KEY"); CLIENT = Together(api_key=API_KEY) if API_KEY else None

def llm(messages, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    if CLIENT is None: raise RuntimeError("TOGETHER_API_KEY not set")
    def call(m):
        r = CLIENT.chat.completions.create(model=m, messages=messages, max_tokens=320, temperature=0.6, top_p=0.9, stream=False)
        return r.choices[0].message.content.strip()
    last=None
    for a in range(4):
        try: return call(model)
        except Exception as e:
            s=str(e); last=s
            if ("429" in s or "RateLimit" in s) and model.endswith("-Turbo"):
                try: return call(model+"-Free")
                except Exception as e2: last=str(e2)
            if any(t in s for t in ["429","temporarily","timeout","Service Unavailable","TLS"]):
                time.sleep((2**a)+random.random()); continue
            break
    raise RuntimeError(last or "model error")

# =================== COACHING PROMPT ===================
def build_system_prompt(include_quote=False):
    # Quote is optional (off by default)
    quote_rule = "Optionally end with a short micro-quote (<= 12 words) if it truly fits." if include_quote else "Do NOT include a quote."
    return (
        "You are a friendly coach chatting like WhatsApp. Be warm, validating, and practical. "
        "Keep replies under 140 words. Structure:\n"
        "1) Acknowledge & validate in 1 line.\n"
        "2) What I’m hearing (paraphrase) in 1 line.\n"
        "3) Try now: one 60-second tool (e.g., 4-4-6 breath, 5-4-3-2-1 grounding, shoulder release).\n"
        "4) Next tiny step: one <5-minute task for today.\n"
        "5) Encouragement: one line.\n"
        f"{quote_rule} No clinical language or crisis instructions unless asked."
    )

PREFACE = {
    "sadness":"I’m here with you. Want to share what happened?",
    "anger":"I can hear the frustration—what sparked it?",
    "fear":"That sounds uneasy—what’s making it feel scary?",
    "joy":"Love that—what sparked it?",
    "love":"That’s heart-warming—tell me more.",
    "optimism":"I like that energy—what feels possible?",
    "disgust":"That felt unpleasant—what part hit hardest?",
    "surprise":"Unexpected things can throw us—what happened?",
    "neutral":"I’m listening—what’s on your mind?",
}

def friendly_lead_in(emo): return PREFACE.get(emo, "I’m here with you—want to tell me what happened?")

def coach_reply(user_text, emotion, include_quote=False, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    sys = build_system_prompt(include_quote)
    msgs = [{"role":"system","content":sys},{"role":"user","content":f"Detected emotion: {emotion}. Journal: {user_text}"}]
    try:
        return llm(msgs, model=model)
    except Exception as e:
        # graceful offline fallback
        return ("I’m with you. It sounds heavy.\n\nTry now: 4-4-6 breathing (inhale 4, hold 4, exhale 6) ×4.\n\n"
                "Next tiny step: set a 5-minute timer for the easiest 1%.\n\nYou’ve got this.  "
                f"[details: {e}]")

# =================== UI ===================
# status (top chips)
api_badge = "✅ API key" if API_KEY else "❌ API key"
model_badge = "✅ Emotion model" if MODEL_OK else "❌ Emotion model"
st.markdown(f'<div class="chips"><span class="chip">{api_badge}</span><span class="chip">{model_badge}</span></div>', unsafe_allow_html=True)

# chat state with a friendly greeting
if "chat" not in st.session_state:
    st.session_state.chat = [("assistant","Hi—how are you feeling today?","neutral")]

# render chat
st.markdown('<div class="chatwrap">', unsafe_allow_html=True)
for role, content, emo in st.session_state.chat:
    cls = "user" if role=="user" else "assistant"
    safe = html.escape(content)
    meta = f'<div class="meta">{EMOJI.get(emo,"😐")} {emo.title() if emo else ""}</div>' if emo else ""
    st.markdown(f'<div class="msg {cls}"><div class="bubble">{safe}{meta}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# input
model_choice = st.selectbox("Model", ["meta-llama/Llama-3.3-70B-Instruct-Turbo","meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"], index=0, label_visibility="collapsed")
prompt = st.chat_input("Write how you feel…")

# handle message
if prompt is not None:
    text = (prompt or "").strip()
    if text:
        st.session_state.chat.append(("user", text, None))
        feed = text if len(text.split())>1 else text + " (please respond kindly with one practical step.)"
        emo, conf, _ = detect_emotion_9(feed)
        st.session_state.chat.append(("assistant", friendly_lead_in(emo), emo))
        st.session_state.chat.append(("assistant", coach_reply(feed, emo, include_quote=False, model=model_choice), emo))
        st.rerun()

st.markdown('<div class="footer">© 2025 • Journaling Coach • Friendly guidance, not medical advice.</div>', unsafe_allow_html=True)
