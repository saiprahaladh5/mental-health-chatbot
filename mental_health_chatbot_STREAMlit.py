import os, json, time, random
from uuid import uuid4
from datetime import datetime, date
from typing import List, Dict, Any
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from together import Together

st.set_page_config(page_title="Daily Mental Health Chatbot", layout="wide")
st.title("🧠 Daily Mental Health Chatbot")

SAVE_CHATS = "chats.json"
SAVE_CHECKINS = "checkins.csv"

CRISIS_PHRASES = [
    "suicide","kill myself","end my life","self harm","self-harm",
    "hurt myself","can't go on","dont want to live","don't want to live","ending it"
]

RESOURCES = (
    "**If you’re in immediate danger, call your local emergency number.**\n\n"
    "- 🇺🇸 **988 Suicide & Crisis Lifeline** — call or text **988**, chat via 988lifeline.org\n"
    "- 🌐 **Find a helpline (global)**: findahelpline.com\n"
)

def _get_api_key():
    try:
        return st.secrets["TOGETHER_API_KEY"]
    except Exception:
        return os.getenv("TOGETHER_API_KEY")

API_KEY = _get_api_key()
if not API_KEY:
    st.error("Missing TOGETHER_API_KEY. Add it in Streamlit Secrets or as an environment variable.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_client(key: str):
    return Together(api_key=key)

client = get_client(API_KEY)

def chat_llm(messages, model: str, max_tokens=180, temperature=0.7, top_p=0.9):
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, stream=False
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            s = str(e)
            if "429" in s or "RateLimit" in s or "temporarily unavailable" in s:
                time.sleep((2**attempt) + random.uniform(0,0.5)); continue
            raise
    raise RuntimeError("Rate limit retries exceeded")

@st.cache_resource(show_spinner=False)
def _load_emotion_model():
    name = "SamLowe/roberta-base-go_emotions"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    id2label = ({int(k): v for k, v in mdl.config.id2label.items()}
                if isinstance(mdl.config.id2label, dict)
                else {i: l for i, l in enumerate(mdl.config.id2label)})
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
    idx_map = {g: [] for g in groups}
    for i in range(mdl.config.num_labels):
        lab = str(id2label[i]).lower()
        placed = False
        for g, fine in groups.items():
            if lab in fine:
                idx_map[g].append(i); placed = True; break
        if not placed:
            idx_map["neutral"].append(i)
    return tok, mdl, idx_map

tokenizer, emotion_model, group_index_map = _load_emotion_model()

def detect_emotion_9(text: str):
    x = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        logits = emotion_model(**x).logits
    p = torch.sigmoid(logits)[0].cpu().numpy()
    scores = {g: float(sum(p[i] for i in idxs)) for g, idxs in group_index_map.items()}
    total = sum(scores.values()) or 1.0
    scores = {k: v/total for k, v in scores.items()}
    top = max(scores, key=scores.get)
    return top, scores[top], scores

def load_chats() -> Dict[str, Any]:
    try:
        if os.path.exists(SAVE_CHATS):
            with open(SAVE_CHATS, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def save_chats():
    try:
        with open(SAVE_CHATS, "w", encoding="utf-8") as f:
            json.dump(st.session_state.chats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Couldn't save chats: {e}")

def append_checkin(row: Dict[str, Any]):
    df = pd.DataFrame([row])
    exists = os.path.exists(SAVE_CHECKINS)
    df.to_csv(SAVE_CHECKINS, mode="a", header=not exists, index=False)

def read_checkins() -> pd.DataFrame:
    if os.path.exists(SAVE_CHECKINS):
        try:
            return pd.read_csv(SAVE_CHECKINS)
        except Exception:
            pass
    return pd.DataFrame(columns=["timestamp","date","mood","emotion","confidence","journal","response"])

def is_crisis(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in CRISIS_PHRASES)

def safe_support_reply(user_text: str, emotion: str, history: List[Dict[str,str]], model: str) -> str:
    if is_crisis(user_text):
        return ("I'm really sorry you're feeling this way. You deserve immediate support. "
                "If you are in danger or considering harming yourself, please contact help right now.\n\n"
                + RESOURCES +
                "\n\nIf you want, we can also talk about one small grounding step (e.g., 4-7-8 breathing, "
                "holding ice, naming five things you see). You're not alone.")
    sys = ("You assist with daily mental health check-ins. Acknowledge the feeling, give EXACTLY one practical step, "
           "and include ONE short verse reference only (e.g., 'Qur’an 94:5', 'Psalm 34:18', 'Bhagavad Gita 2.47'). "
           "Keep the reply under 100 words. Be warm, specific, and non-clinical.")
    msgs = [{"role":"system","content": sys}] + history[-4:] + [{"role":"user","content": f"Emotion: {emotion}\nJournal: {user_text}"}]
    return chat_llm(msgs, model=model, max_tokens=180, temperature=0.7, top_p=0.9)

with st.sidebar:
    st.subheader("Settings")
    model_choice = st.selectbox(
        "Model",
        ["meta-llama/Llama-3.3-70B-Instruct-Turbo","meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"],
        index=0,
    )

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "current_chat_id" not in st.session_state:
    if st.session_state.chats:
        st.session_state.current_chat_id = next(iter(st.session_state.chats))
    else:
        cid = str(uuid4())
        st.session_state.chats[cid] = {"title": "Chat 1","created_at": datetime.utcnow().isoformat(),"messages": []}
        st.session_state.current_chat_id = cid

def current_chat() -> Dict[str, Any]:
    return st.session_state.chats[st.session_state.current_chat_id]

with st.sidebar:
    st.subheader("Conversations")
    chat_ids = list(st.session_state.chats.keys())
    def _fmt(cid): return st.session_state.chats[cid].get("title", cid)
    idx = chat_ids.index(st.session_state.current_chat_id)
    selected = st.selectbox("Select chat", options=chat_ids, index=idx, format_func=_fmt)
    if selected != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected
        st.experimental_rerun()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New chat"):
            cid = str(uuid4())
            st.session_state.chats[cid] = {"title": f"Chat {len(st.session_state.chats)+1}","created_at": datetime.utcnow().isoformat(),"messages": []}
            st.session_state.current_chat_id = cid
            save_chats()
            st.experimental_rerun()
    with col2:
        if st.button("🗑️ Delete chat"):
            del st.session_state.chats[st.session_state.current_chat_id]
            if st.session_state.chats:
                st.session_state.current_chat_id = next(iter(st.session_state.chats))
            else:
                cid = str(uuid4())
                st.session_state.chats[cid] = {"title": "Chat 1","created_at": datetime.utcnow().isoformat(),"messages": []}
                st.session_state.current_chat_id = cid
            save_chats()
            st.experimental_rerun()
    new_title = st.text_input("Rename current chat", value=current_chat().get("title","Untitled"))
    if new_title and new_title != current_chat().get("title"):
        current_chat()["title"] = new_title
        save_chats()

tab1, tab2, tab3 = st.tabs(["🌅 Daily Check-in","💬 Open Chat","🗂️ History"])

with tab1:
    st.subheader("Daily Check-in")
    mood = st.select_slider("How are you feeling right now?", options=["😢","🙁","😐","🙂","😀"], value="😐")
    journal = st.text_area("What's on your mind today?", height=140, placeholder="Type a few sentences...")
    c1, c2, _ = st.columns([1,1,2])
    with c1:
        do_reply = st.button("Start conversation", type="primary")
    with c2:
        do_clear = st.button("Clear")
    if do_clear:
        st.experimental_rerun()
    if do_reply and journal.strip():
        emo, conf, scores = detect_emotion_9(journal)
        history = [{"role": m["role"], "content": m["content"]} for m in current_chat()["messages"][-6:]]
        reply = safe_support_reply(journal, emo, history, model_choice)
        with st.chat_message("user"):
            st.markdown(journal)
        with st.chat_message("assistant"):
            st.markdown(f"**Detected emotion:** {emo} ({conf:.2f})\n\n{reply}")
        current_chat()["messages"].append({"role": "user", "content": journal})
        current_chat()["messages"].append({"role": "assistant", "content": f"Detected emotion: {emo} ({conf:.2f})\n\n{reply}"})
        save_chats()
        append_checkin({
            "timestamp": datetime.utcnow().isoformat(),
            "date": date.today().isoformat(),
            "mood": mood,
            "emotion": emo,
            "confidence": round(conf, 4),
            "journal": journal,
            "response": reply,
        })
        with st.expander("Show emotion scores"):
            st.dataframe(pd.DataFrame.from_dict(scores, orient="index", columns=["score"]).sort_values("score", ascending=False))

with tab2:
    st.subheader(current_chat().get("title","Chat"))
    for m in current_chat()["messages"]:
        with st.chat_message("user" if m["role"]=="user" else "assistant"):
            st.markdown(m["content"])
    if not current_chat()["messages"]:
        st.info("Try a quick starter: “I feel anxious about tomorrow’s meeting.”, “Today I’m grateful for…”, or “I can’t focus lately.”")
    user_prompt = st.chat_input("Type your message")
    if user_prompt:
        current_chat()["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        emo, conf, _ = detect_emotion_9(user_prompt)
        hist = [{"role": m["role"], "content": m["content"]} for m in current_chat()["messages"][-6:]]
        reply = safe_support_reply(user_prompt, emo, hist, model_choice)
        assistant_text = f"Detected emotion: {emo} ({conf:.2f})\n\n{reply}"
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
        current_chat()["messages"].append({"role": "assistant", "content": assistant_text})
        save_chats()

with tab3:
    st.subheader("Journal History")
    df = read_checkins()
    if df.empty:
        st.write("No check-ins yet. Use the Daily Check-in tab to start.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            date_filter = st.date_input("Filter by date", value=None)
        with col_b:
            emotion_filter = st.multiselect("Filter by emotion", options=sorted(df["emotion"].unique()))
        view = df.copy()
        if date_filter:
            view = view[view["date"] == date_filter.isoformat()]
        if emotion_filter:
            view = view[view["emotion"].isin(emotion_filter)]
        view = view.sort_values("timestamp", ascending=False)
        st.dataframe(view[["date","mood","emotion","confidence","journal","response"]], use_container_width=True, height=420)

st.markdown(
    "<hr><small><em>This chatbot provides supportive conversation, not medical advice. "
    "If you’re in crisis or thinking of self-harm, please use the resources above and seek immediate help.</em></small>",
    unsafe_allow_html=True,
)
