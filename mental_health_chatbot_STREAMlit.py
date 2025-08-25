import os, time, random, html
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import expit
from together import Together

st.set_page_config(page_title="Journaling Coach (Chat)", layout="centered")
st.markdown("""
<style>
:root { --bg:#0b1220; --surface:#0f172a; --border:#1f2937; --txt:#e2e8f0; --muted:#94a3b8; --accent:#22c55e; }
html, body .main { background:var(--bg) !important; color:var(--txt) !important; }
.block-container { padding-top: 1.4rem; max-width: 860px; }
.header { background:linear-gradient(135deg,#0ea5e9 0%,#22c55e 100%); border-radius:16px; padding:14px 16px; color:#062332; font-weight:700; margin-bottom:12px; }
.chat { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding: 8px 10px; min-height: 360px; }
.msg { display:flex; margin:12px 6px; }
.msg .bubble { border:1px solid var(--border); border-radius:18px; padding:10px 14px; line-height:1.45; max-width: 86%; }
.msg.user { justify-content:flex-end; }
.msg.user .bubble { background:#0b254d; border-color:#153e7a; }
.msg.assistant .bubble { background:#111827; }
.meta { font-size:.78rem; color:var(--muted); margin-top:4px }
.tools { display:flex; gap:.5rem; flex-wrap:wrap; margin:8px 0 0 }
.tool { padding:.35rem .6rem; border-radius:999px; background:#0b1325; border:1px solid #1f2937; color:#cbd5e1; font-size:.9rem }
.pill { display:inline-flex; align-items:center; gap:.4rem; border:1px solid var(--border); padding:.32rem .6rem; border-radius:999px; margin:.25rem .25rem 0 0; background:#0b1325; color:#cbd5e1; font-size:.92rem; }
.small { color:var(--muted); font-size:.9rem }
hr { border:none; height:1px; background:#1f2937; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header">🧠 Journaling Coach — warm, practical, emotion-aware</div>', unsafe_allow_html=True)

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
        lab = id2label[i].lower()
        placed=False
        for g,fine in groups.items():
            if lab in fine: idx[g].append(i); placed=True; break
        if not placed: idx["neutral"].append(i)
    return tok, mdl, idx

try:
    TOKENIZER, EMO_MODEL, IDX = _load_emotion_model(); MODEL_OK=True
except Exception: TOKENIZER=EMO_MODEL=IDX=None; MODEL_OK=False

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

API_KEY = os.getenv("TOGETHER_API_KEY")
CLIENT  = Together(api_key=API_KEY) if API_KEY else None

def _chat(messages, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    if CLIENT is None: raise RuntimeError("TOGETHER_API_KEY not set")
    def _call(m):
        r = CLIENT.chat.completions.create(model=m, messages=messages, max_tokens=320, temperature=0.6, top_p=0.9, stream=False)
        return r.choices[0].message.content.strip()
    last=None
    for a in range(4):
        try: return _call(model)
        except Exception as e:
            s=str(e); last=s
            if ("429" in s or "RateLimit" in s) and model.endswith("-Turbo"):
                try: return _call(model+"-Free")
                except Exception as e2: last=str(e2)
            if any(t in s for t in ["429","temporarily","timeout","Service Unavailable"]):
                time.sleep((2**a)+random.random()); continue
            break
    raise RuntimeError(last or "model error")

STYLE_RULES = {
    "Coach": "Tone warm, validating, down-to-earth. Encourage tiny, doable steps. No therapy jargon.",
    "Cheerleader": "Upbeat, supportive, confidence-building. Celebrate small wins. Keep it concise.",
    "Therapist-lite": "Calm, reflective, non-judgmental. Use gentle CBT/DBT hints without labels.",
}
FOCUS_RULES = {
    "Regulate emotion": "Offer one 60-second regulation tool (e.g., 4-4-6 breath, 5-4-3-2-1 grounding, progressive release).",
    "Reframe thought": "Offer one simple reframe or alternative perspective; ask a curious question.",
    "Plan next step": "Offer one tiny, concrete step that can be done in <5 minutes.",
}

def build_system_prompt(style:str, focus:str, include_quote:bool):
    quote_rule = "End with one short micro-quote (<= 12 words) relevant to the feeling." if include_quote else "Do NOT include any quote."
    return (
        "You are a friendly journaling coach. "
        f"{STYLE_RULES[style]} {FOCUS_RULES[focus]} {quote_rule} "
        "Structure your reply in 5 short parts:\n"
        "1) Acknowledge & validate in 1 sentence.\n"
        "2) What I’m hearing (paraphrase the core concern).\n"
        "3) Try now (one actionable 60-sec exercise, bullet or numbered).\n"
        "4) Next tiny step (one concrete task the user can do today).\n"
        "5) Encouragement (one line). "
        "Stay under 150 words. No emojis. No clinical or crisis instructions unless requested."
    )

def gen_reply(user_text:str, emotion:str, style:str, focus:str, include_quote:bool, model:str):
    sys = build_system_prompt(style, focus, include_quote)
    msgs = [
        {"role":"system","content": sys},
        {"role":"user","content": f"Detected emotion: {emotion}. Journal: {user_text}"},
    ]
    try:
        return _chat(msgs, model=model)
    except Exception as e:
        # Friendly offline fallback with real guidance
        fallback = {
            "Regulate emotion": "Try 4-4-6 breathing: inhale 4, hold 4, exhale 6 — repeat 4x.",
            "Reframe thought": "Ask: ‘What would I tell a friend in my shoes?’ Write one kinder line.",
            "Plan next step": "Set a 5-minute timer and do the easiest 1% (one email or one line).",
        }[focus]
        ack = f"I hear {emotion} in what you shared. You're not alone."
        paraphrase = "It sounds like there’s a lot on your mind and it feels heavy right now."
        encourage = "Small steps count—consistency beats intensity."
        tail = "Short line to carry: “This moment will move.”" if include_quote else ""
        return f"{ack}\n\nWhat I’m hearing: {paraphrase}\n\nTry now: {fallback}\n\nNext tiny step: Pick a 5-minute task and start.\n\n{encourage}\n{tail}  \n[details: {e}]"

# ---------- Sidebar controls ----------
with st.sidebar:
    st.markdown("### Response settings")
    STYLE = st.selectbox("Tone", ["Coach","Cheerleader","Therapist-lite"], index=0)
    FOCUS = st.selectbox("Focus", ["Regulate emotion","Reframe thought","Plan next step"], index=0)
    INCLUDE_QUOTE = st.checkbox("Include short quote", value=False)
    MODEL = st.selectbox("Model", ["meta-llama/Llama-3.3-70B-Instruct-Turbo","meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"], index=0)
    st.markdown("### Status")
    st.caption(("✅ API key found" if API_KEY else "❌ API key missing") + " · " + ("✅ Model loaded" if MODEL_OK else "❌ Model failed"))

# ---------- Emotion palette ----------
st.markdown("#### Emotions")
for k,v in EMOJI.items():
    st.markdown(f"<span class='pill'>{v} {k.title()}</span>", unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ---------- Chat state ----------
if "chat" not in st.session_state: st.session_state.chat=[]
if "last_input" not in st.session_state: st.session_state.last_input=""
if "last_emotion" not in st.session_state: st.session_state.last_emotion=None
if "style" not in st.session_state: st.session_state.style="Coach"
if "focus" not in st.session_state: st.session_state.focus="Regulate emotion"
if "q" not in st.session_state: st.session_state.q=False

# Render conversation
st.markdown('<div class="chat">', unsafe_allow_html=True)
for role, content, emo in st.session_state.chat:
    safe = html.escape(content)
    if role=="user":
        st.markdown(f"<div class='msg user'><div class='bubble'>{safe}</div></div>", unsafe_allow_html=True)
    else:
        emo_tag = f"<div class='meta'>{EMOJI.get(emo,'😐')} {emo.title() if emo else ''}</div>" if emo else ""
        st.markdown(f"<div class='msg assistant'><div class='bubble'>{safe}{emo_tag}</div></div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input + actions
prompt = st.chat_input("Write how you feel…")
try_another = st.button("Try another angle")

def respond_to(text:str, style:str, focus:str, include_quote:bool):
    feed = text if len(text.split())>1 else text + " I'm checking in—please respond kindly with one practical step."
    emo, conf, _ = detect_emotion_9(feed)
    reply = gen_reply(feed, emo, style, focus, include_quote, MODEL)
    st.session_state.last_emotion = emo
    st.session_state.chat.append(("assistant", reply, emo))

if prompt is not None and prompt.strip():
    st.session_state.chat.append(("user", prompt.strip(), None))
    st.session_state.last_input = prompt.strip()
    respond_to(prompt.strip(), STYLE, FOCUS, INCLUDE_QUOTE)
    st.rerun()

if try_another and st.session_state.last_input:
    respond_to(st.session_state.last_input, STYLE, FOCUS, INCLUDE_QUOTE)
    st.rerun()

# Footer
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("© 2025 • Journaling Coach • Friendly guidance, not medical advice. If you’re in crisis, contact local emergency services.")
