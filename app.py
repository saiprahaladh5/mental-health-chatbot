import os, re, time, random, html
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import expit
from together import Together

# ===== UI =====
st.set_page_config(page_title="Companion • Coach • Assistant", layout="centered")
st.markdown("""
<style>
:root{--bg:#0b1220;--panel:#0f172a;--border:#1f2937;--text:#e5e7eb;--muted:#94a3b8;--in:#101827;--out:#2563eb;--outb:#1e40af}
html, body .main { background:var(--bg)!important; color:var(--text)!important }
.block-container{max-width:900px;padding-top:10px}
.appbar{background:linear-gradient(135deg,#0ea5e9 0%,#22c55e 100%);color:#062332;border-radius:18px;padding:12px 16px;font-weight:700;box-shadow:0 8px 30px rgba(0,0,0,.25);margin-bottom:12px}
.chat{background:var(--panel);border:1px solid var(--border);border-radius:18px;padding:10px 12px;min-height:420px;box-shadow:0 8px 40px rgba(0,0,0,.25)}
.msg{display:flex;margin:10px 6px}
.bubble{border-radius:16px;padding:12px 14px;line-height:1.5;max-width:78%;border:1px solid var(--border)}
.assistant .bubble{background:var(--in)}
.user{justify-content:flex-end}
.user .bubble{background:var(--out);color:#fff;border-color:var(--outb);box-shadow:0 4px 18px rgba(37,99,235,.35)}
.meta{font-size:.78rem;color:#94a3b8;margin-top:6px}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="appbar">🤝 Friend-Coach vibe • No scripts • Holy line only for emotions</div>', unsafe_allow_html=True)

EMOJI = {"joy":"😊","sadness":"😢","anger":"😠","fear":"😟","disgust":"🤢","surprise":"😮","love":"💖","optimism":"🌤️","neutral":"😐"}

# ===== Emotion model (coach only) =====
@st.cache_resource(show_spinner=False)
def _load_emotion_model():
    name = "SamLowe/roberta-base-go_emotions"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
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

# ===== LLM =====
API_KEY = os.getenv("TOGETHER_API_KEY")
if not API_KEY:
    st.error("TOGETHER_API_KEY missing."); st.stop()
CLIENT = Together(api_key=API_KEY)

def llm(messages, model="meta-llama/Llama-3.3-70B-Instruct-Turbo", max_tokens=520, temperature=0.55, top_p=0.9):
    def call(m):
        r = CLIENT.chat.completions.create(model=m, messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=False)
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

# ===== Intents (task first, feelings last) =====
GREETINGS = {"hi","hello","hey","hola","yo","sup","hai","hii","hi there","hello there"}
TRAVEL = {"travel","trip","vacation","holiday","destination","destinations","place","places","country","city","visit","itinerary","spots","spot"}
LIST_WORDS = {"top","best","list","recommend","suggest","ideas","options","give me","show me","list down"}
CODE   = {"python","code","bug","error","function","class","api","sql","javascript","react","streamlit"}
COMPARE = {" vs ","versus","difference between","compare","against"}

def is_greeting(t:str)->bool: return t.strip().lower() in GREETINGS
def has_any(text, vocab): return any(w in text.lower() for w in vocab)
def extract_topn(text, default=5):
    m = re.search(r"\btop\s*(\d{1,2})\b", text.lower())
    if m:
        try: return max(1, min(20, int(m.group(1))))
        except: pass
    return default

def detect_intent(text:str)->str:
    t=text.lower()
    if is_greeting(t): return "greeting"
    if has_any(t, TRAVEL) and has_any(t, LIST_WORDS): return "travel_list"
    if has_any(t, TRAVEL): return "travel_general"
    if has_any(t, CODE): return "coding"
    if has_any(t, COMPARE): return "compare"
    if has_any(t, LIST_WORDS): return "general_list"
    if any(k in t for k in ["i feel","i'm feeling","im feeling","sad","lonely","anxious","depressed","stress","stressed","overwhelmed","angry","confused","frustrated","heartbroken","afraid","scared"]):
        return "coach"
    return "general"

# ===== Holy micro-quote (coach only) =====
def holy_snippet(emotion:str):
    e=(emotion or "neutral").lower()
    if e in {"sadness","grief"}: return '“The LORD is near to the brokenhearted.” (Psalm 34:18)'
    if e in {"fear","nervousness","confusion"}: return '“With hardship comes ease.” (Qur’an 94:5–6)'
    if e in {"anger"}: return '“A gentle answer turns away wrath.” (Proverbs 15:1)'
    if e in {"love"}: return '“Love is patient and kind.” (1 Corinthians 13:4)'
    if e in {"optimism","joy","relief","gratitude"}: return '“Be joyful in hope.” (Romans 12:12)'
    return '“Be steadfast in yoga.” (Bhagavad-Gita 2:48)'

# ===== HARD BAN in coach (kills clichéd scripts) =====
BANNED = [
    r"(?i)\btry\s*now\b", r"(?i)\bnext\s*tiny\s*step\b",
    r"(?i)\b4[-\s]?4[-\s]?6\b", r"(?i)\bbreath(e|ing|work)?\b", r"(?i)\binhale\b", r"(?i)\bexhale\b",
    r"(?i)\bdeep\s*breath\b", r"(?i)\bground(ing|ed)\b", r"(?i)\bgratitude\b",
    r"(?i)\bwrite\s*down\s*(three|3)\b", r"(?i)\bpause\b", r"(?i)\bsavor\b", r"(?i)\breflect\b",
    r"(?i)\btake\s*(\d+|\w+)\s*(seconds?|minutes?)\b",
]
def sanitize(text:str)->str:
    lines = [ln for ln in text.splitlines() if not any(re.search(p, ln) for p in BANNED)]
    out = "\n".join(lines).strip()
    for p in BANNED: out = re.sub(p, "", out)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", out).strip()

# ===== Prompts =====
def sys_greeting():  return "One-line greeting and ask what they want help with. 10 words max. No emojis."
def sys_travel_list(n): return (f"Return exactly {n} items as a markdown list. Each: **City, Country** — 1-line why | Best season | Vibe | Budget ($/$$/$$$). No intro/outro. No coaching.")
def sys_travel_general(): return "Concrete places, seasons, and a sample 3-day outline. No quotes. No coaching."
def sys_coding():  return "Senior engineer. Minimal runnable example first, then brief explanation + pitfalls. No filler."
def sys_compare(): return "Compact table: Option | Summary | Pros | Cons | Best for. Then a decisive verdict."
def sys_general_list(n): return f"Return exactly {n} punchy one-line bullets tailored to the request. No intro/outro."
def sys_general(): return "Answer directly. Use bullets/steps/table when helpful. Be specific."
def sys_coach_convo(): 
    return ("Trusted friend & coach. One short paragraph (70–110 words), natural and human. "
            "Validate what they said, reflect it in your own words, and offer 1–2 concrete ideas that fit their situation. "
            "Absolutely avoid scripts, breathwork, counting, 'try now', journaling prompts, or gratitude checklists. "
            "No lists unless the user asks for steps. No quotes (a separate holy line will be appended).")

# ===== State & render =====
if "chat" not in st.session_state:
    st.session_state.chat = [("assistant","Hey — I’m here. What do you want help with today?","neutral")]

colA, colB = st.columns([1,1])
mode = colA.selectbox("Mode", ["Auto (router)", "Companion", "Coach"], index=0)
model_choice = colB.selectbox("Model", ["meta-llama/Llama-3.3-70B-Instruct-Turbo","meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"], index=0)

st.markdown('<div class="chat">', unsafe_allow_html=True)
for role, content, emo in st.session_state.chat:
    cls = "user" if role=="user" else "assistant"
    safe = html.escape(content)
    meta = f"<div class='meta'>{EMOJI.get(emo,'')+' '+(emo.title() if emo else '')}</div>" if emo else ""
    st.markdown(f"<div class='msg {cls}'><div class='bubble'>{safe}{meta}</div></div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

prompt = st.chat_input("Talk like with a friend. Ask tasks (travel, coding) or share how you feel…")

def respond(text:str):
    t = text.strip()

    # Companion: warm chat (no therapy)
    if mode == "Companion":
        sys = ("Warm, natural companion. Keep replies under 90 words. Be curious and supportive. "
               "No therapy instructions, no breathwork, no quotes, no holy references. "
               "If the user asks for tasks, answer directly and concise.")
        msgs = [{"role":"system","content":sys}] + [{"role":"user","content":t}]
        st.session_state.chat.append(("assistant", llm(msgs, model=model_choice, max_tokens=360, temperature=0.65), None))
        return

    # Coach: conversational + holy line; sanitized
    if mode == "Coach":
        emo, _, _ = detect_emotion_9(t)
        msgs = [{"role":"system","content":sys_coach_convo()}, {"role":"user","content": t + f" [emotion: {emo}]"}]
        raw = llm(msgs, model=model_choice, max_tokens=420, temperature=0.5)
        clean = sanitize(raw)
        holy = holy_snippet(emo)
        if holy: clean = clean.strip() + f"\n\n— {holy}"
        st.session_state.chat.append(("assistant", clean, emo))
        return

    # Auto (router)
    intent = detect_intent(t)
    if intent == "greeting":
        st.session_state.chat.append(("assistant", llm([{"role":"system","content":sys_greeting()}, {"role":"user","content":t}], model=model_choice, max_tokens=32, temperature=0.2), None))
    elif intent == "travel_list":
        n = extract_topn(t, 5)
        st.session_state.chat.append(("assistant", llm([{"role":"system","content":sys_travel_list(n)}, {"role":"user","content":t}], model=model_choice, max_tokens=340, temperature=0.35), None))
    elif intent == "travel_general":
        st.session_state.chat.append(("assistant", llm([{"role":"system","content":sys_travel_general()}, {"role":"user","content":t}], model=model_choice, max_tokens=420, temperature=0.45), None))
    elif intent == "coding":
        st.session_state.chat.append(("assistant", llm([{"role":"system","content":sys_coding()}, {"role":"user","content":t}], model=model_choice, max_tokens=580, temperature=0.3), None))
    elif intent == "compare":
        st.session_state.chat.append(("assistant", llm([{"role":"system","content":sys_compare()}, {"role":"user","content":t}], model=model_choice, max_tokens=420, temperature=0.35), None))
    elif intent == "general_list":
        n = extract_topn(t, 5)
        st.session_state.chat.append(("assistant", llm([{"role":"system","content":sys_general_list(n)}, {"role":"user","content":t}], model=model_choice, max_tokens=300, temperature=0.35), None))
    elif intent == "coach":
        emo, _, _ = detect_emotion_9(t)
        raw = llm([{"role":"system","content":sys_coach_convo()}, {"role":"user","content": t + f" [emotion: {emo}]"}], model=model_choice, max_tokens=420, temperature=0.5)
        clean = sanitize(raw)
        holy = holy_snippet(emo)
        if holy: clean = clean.strip() + f"\n\n— {holy}"
        st.session_state.chat.append(("assistant", clean, emo))
    else:
        st.session_state.chat.append(("assistant", llm([{"role":"system","content":sys_general()}, {"role":"user","content":t}], model=model_choice, max_tokens=460, temperature=0.5), None))

if prompt is not None:
    text = (prompt or "").strip()
    if text:
        st.session_state.chat.append(("user", text, None))
        respond(text)
        st.rerun()
