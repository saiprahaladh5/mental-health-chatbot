import os, re, time, random, html
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import expit
from together import Together

# ====== PAGE + THEME ======
st.set_page_config(page_title="Turbo Assistant", layout="centered")
st.markdown("""
<style>
:root{--bg:#0b1220;--panel:#0f172a;--border:#1f2937;--text:#e5e7eb;--muted:#94a3b8;
      --in:#101827;--out:#2563eb;--outb:#1e40af}
html, body .main { background:var(--bg)!important; color:var(--text)!important }
.block-container{max-width:860px;padding-top:10px}
.appbar{background:linear-gradient(135deg,#0ea5e9 0%,#22c55e 100%);color:#062332;border-radius:18px;padding:12px 16px;font-weight:700;box-shadow:0 8px 30px rgba(0,0,0,.25);margin-bottom:12px}
.chips{display:flex;gap:.4rem;flex-wrap:wrap;margin:6px 0 12px}.chip{padding:.34rem .6rem;border-radius:999px;border:1px solid var(--border);color:#cbd5e1;background:#0b1325;font-size:.9rem}
.chat{background:var(--panel);border:1px solid var(--border);border-radius:18px;padding:10px 12px;min-height:460px;box-shadow:0 8px 40px rgba(0,0,0,.25)}
.msg{display:flex;margin:12px 6px}.bubble{border-radius:16px;padding:12px 14px;line-height:1.5;max-width:80%;border:1px solid var(--border)}
.assistant .bubble{background:var(--in)}.user{justify-content:flex-end}.user .bubble{background:var(--out);color:#fff;border-color:var(--outb);box-shadow:0 4px 18px rgba(37,99,235,.35)}
.meta{font-size:.78rem;color:var(--muted);margin-top:6px}.footer{color:var(--muted);margin-top:10px}
hr{border:none;height:1px;background:var(--border)}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="appbar">⚡ Turbo Assistant — intent aware, crisp & helpful</div>', unsafe_allow_html=True)

# ====== EMOTION MODEL (for coach mode only) ======
EMOJI = {"joy":"😊","sadness":"😢","anger":"😠","fear":"😟","disgust":"🤢","surprise":"😮","love":"💖","optimism":"🌤️","neutral":"😐"}

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

# ====== TOGETHER CLIENT ======
API_KEY = os.getenv("TOGETHER_API_KEY")
if not API_KEY:
    st.error("TOGETHER_API_KEY missing.")
    st.stop()
CLIENT = Together(api_key=API_KEY)

def llm(messages, model="meta-llama/Llama-3.3-70B-Instruct-Turbo", max_tokens=600, temperature=0.6, top_p=0.9):
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

# ====== INTENT ROUTER ======
TRAVEL = {"travel","trip","vacation","holiday","destination","destinations","place","places","country","city","visit","itinerary","spots"}
CODE   = {"python","code","bug","error","function","class","api","sql","javascript","react","streamlit"}
COMPARE_WORDS = {"vs","versus","difference between","compare"}
LIST_WORDS = {"top","best","list","recommend","suggest","ideas","options"}
FEEL_WORDS = {"i feel","i'm feeling","im feeling","sad","lonely","anxious","anxiety","depressed","stress","stressed","overwhelmed","angry","heartbroken","burned out","tired of"}

def get_topn(text, default=5):
    m = re.search(r"\btop\s*(\d{1,3})\b", text.lower())
    if m: 
        try: return max(1, min(50, int(m.group(1))))
        except: pass
    m2 = re.search(r"\b(\d{1,2})\s*(?:places|items|ideas|options)\b", text.lower())
    if m2:
        try: return max(1, min(50, int(m2.group(1))))
        except: pass
    return default

def contains_any(text, vocab):
    t = text.lower()
    return any(w in t for w in vocab)

def detect_intent(text:str)->str:
    t = text.lower()
    if contains_any(t, FEEL_WORDS): return "coach"
    if contains_any(t, CODE): return "coding"
    if contains_any(t, TRAVEL) and contains_any(t, LIST_WORDS): return "travel_list"
    if contains_any(t, TRAVEL): return "travel_general"
    if contains_any(t, COMPARE_WORDS): return "compare"
    if contains_any(t, LIST_WORDS): return "general_list"
    # lightweight fallback: let LLM decide when ambiguous
    try:
        sys = "Classify the user's intent into one of: coach, coding, travel_list, travel_general, compare, general_list, general. Reply with just the label."
        intent = llm([{"role":"system","content":sys},{"role":"user","content":text}], max_tokens=5, temperature=0.0)
        intent = intent.strip().lower()
        return intent if intent in {"coach","coding","travel_list","travel_general","compare","general_list","general"} else "general"
    except Exception:
        return "general"

# ====== SYSTEM PROMPTS ======
def sys_coach():
    return ("You are a friendly journaling coach. Be warm and practical. Keep under 140 words.\n"
            "Structure: 1) Acknowledge  2) What I’m hearing  3) Try now (60s tool)  4) Next tiny step (<5m)  5) Encouragement.\n"
            "No quotes unless asked. No clinical/crisis advice unless requested.")

def sys_travel_list(n):
    return (f"You are a precise travel expert. Return exactly {n} items as a markdown list. For each item include:\n"
            "• **City, Country** — one-line why it’s special\n"
            "• Best season/months\n"
            "• Vibe (culture / nature / nightlife / relax / adventure)\n"
            "• Budget: $, $$, $$$\n"
            "No preamble, no afterword. Keep tight and factual.")

def sys_general_list(n):
    return (f"Return exactly {n} concise bullet items tailored to the request. Each bullet should be one line, punchy, and non-repetitive. No intro/outro.")

def sys_coding():
    return ("Act as a senior engineer. Give a minimal, runnable solution first, then a short explanation and pitfalls. "
            "If the user mentions a file or error, show the fix diff or the minimal repro. No filler. Use Python unless specified.")

def sys_compare():
    return ("Make a clear comparison. Output a small markdown table with rows as options and columns: Summary • Pros • Cons • Best for. "
            "Then one-paragraph verdict. Be decisive.")

def sys_general():
    return ("You are a world-class general assistant. Answer directly and helpfully. Use structure (bullets, steps, or a tiny table) when it improves clarity. "
            "Ask one brief clarifying question only if absolutely required; otherwise just answer.")

# ====== HISTORY BUILDER ======
def build_history(limit=10):
    msgs=[]
    # keep recent conversation to maintain context
    for role, content, _emo in st.session_state.chat[-limit:]:
        msgs.append({"role": role, "content": content})
    return msgs

# ====== CHAT STATE ======
if "chat" not in st.session_state:
    st.session_state.chat = [("assistant","Hi—how can I help today?","neutral")]

# ====== STATUS CHIPS ======
st.markdown(f"<div class='chips'><span class='chip'>✅ Together API</span><span class='chip'>{'✅ Emotion model' if MODEL_OK else '⚠️ Emotion model off'}</span></div>", unsafe_allow_html=True)

# ====== RENDER CHAT ======
st.markdown('<div class="chat">', unsafe_allow_html=True)
for role, content, emo in st.session_state.chat:
    cls = "user" if role=="user" else "assistant"
    safe = html.escape(content)
    meta = f"<div class='meta'>{EMOJI.get(emo,'')+' '+(emo.title() if emo else '')}</div>" if emo else ""
    st.markdown(f"<div class='msg {cls}'><div class='bubble'>{safe}{meta}</div></div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ====== INPUT ======
model_choice = st.selectbox("Model", ["meta-llama/Llama-3.3-70B-Instruct-Turbo","meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"], index=0, label_visibility="collapsed")
prompt = st.chat_input("Ask anything… (e.g., 'top 5 places to visit', 'fix this Python error', 'I'm feeling overwhelmed')")

# ====== HANDLE MESSAGE ======
if prompt is not None:
    text = (prompt or "").strip()
    if text:
        st.session_state.chat.append(("user", text, None))
        intent = detect_intent(text)
        n = get_topn(text, 5)

        try:
            if intent == "coach":
                emo, _, _ = detect_emotion_9(text)
                sys = sys_coach()
                msgs = [{"role":"system","content":sys}] + build_history() + [{"role":"user","content": text + f"\n\n[emotion hint: {emo}]"}]
                reply = llm(msgs, model=model_choice, max_tokens=450)
                st.session_state.chat.append(("assistant", reply, emo))
            elif intent == "travel_list":
                sys = sys_travel_list(n)
                msgs = [{"role":"system","content":sys}, {"role":"user","content": text}]
                reply = llm(msgs, model=model_choice, max_tokens=350, temperature=0.4)
                st.session_state.chat.append(("assistant", reply, None))
            elif intent == "travel_general":
                sys = sys_general()
                msgs = [{"role":"system","content":sys}] + build_history() + [{"role":"user","content": text + "\nFocus on concrete places, seasons, and sample 3-day plan."}]
                reply = llm(msgs, model=model_choice, max_tokens=450)
                st.session_state.chat.append(("assistant", reply, None))
            elif intent == "coding":
                sys = sys_coding()
                msgs = [{"role":"system","content":sys}] + build_history() + [{"role":"user","content": text}]
                reply = llm(msgs, model=model_choice, max_tokens=600, temperature=0.3)
                st.session_state.chat.append(("assistant", reply, None))
            elif intent == "compare":
                sys = sys_compare()
                msgs = [{"role":"system","content":sys}, {"role":"user","content": text}]
                reply = llm(msgs, model=model_choice, max_tokens=450, temperature=0.4)
                st.session_state.chat.append(("assistant", reply, None))
            elif intent == "general_list":
                sys = sys_general_list(n)
                msgs = [{"role":"system","content":sys}, {"role":"user","content": text}]
                reply = llm(msgs, model=model_choice, max_tokens=350, temperature=0.4)
                st.session_state.chat.append(("assistant", reply, None))
            else:
                sys = sys_general()
                msgs = [{"role":"system","content":sys}] + build_history() + [{"role":"user","content": text}]
                reply = llm(msgs, model=model_choice, max_tokens=500)
                st.session_state.chat.append(("assistant", reply, None))
        except Exception as e:
            st.session_state.chat.append(("assistant", f"Sorry, I hit a hiccup: {e}", None))

        st.rerun()

# ====== FOOTER ======
st.markdown('<div class="footer">© 2025 • Turbo Assistant • Helpful answers, concise & structured.</div>', unsafe_allow_html=True)
