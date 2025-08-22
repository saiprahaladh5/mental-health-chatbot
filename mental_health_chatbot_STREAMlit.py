import os, time, random
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import expit
from together import Together

st.set_page_config(page_title="Mental Health Journaling Chatbot", layout="centered")
st.title("Mental Health Journaling Chatbot")

@st.cache_resource(show_spinner=False)
def _load_emotion_model():
    name = "SamLowe/roberta-base-go_emotions"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    id2label = {int(k): v for k, v in mdl.config.id2label.items()} if isinstance(mdl.config.id2label, dict) else mdl.config.id2label

    groups = {
        "joy":       {"amusement","excitement","joy","pride","relief","gratitude"},
        "sadness":   {"sadness","grief","disappointment","remorse"},
        "anger":     {"anger","annoyance","disapproval"},
        "fear":      {"fear","nervousness","embarrassment","confusion"},
        "disgust":   {"disgust"},
        "surprise":  {"surprise","realization"},
        "love":      {"love","caring","admiration","approval","desire"},
        "optimism":  {"optimism","curiosity"},
        "neutral":   {"neutral"},
    }

    idx_map = {g: [] for g in groups}
    for i in range(mdl.config.num_labels):
        lab = id2label[i].lower()
        for g, fine in groups.items():
            if lab in fine:
                idx_map[g].append(i)
                break
        if not any(i in idx_map[g] for g in idx_map):
            idx_map["neutral"].append(i)

    return tok, mdl, idx_map, list(idx_map.keys())

tokenizer, emotion_model, group_index_map, macro_labels = _load_emotion_model()

def detect_emotion_9(text: str):
    x = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = emotion_model(**x).logits
    p = expit(logits.detach().cpu().numpy()[0])

    scores = {}
    for g, idxs in group_index_map.items():
        scores[g] = float(sum(p[i] for i in idxs))
    total = sum(scores.values()) or 1.0
    scores = {k: v/total for k, v in scores.items()}
    top = max(scores, key=scores.get)
    return top, scores[top], scores

def _get_api_key():
    try:
        return st.secrets["TOGETHER_API_KEY"]
    except Exception:
        return os.getenv("TOGETHER_API_KEY")

API_KEY = _get_api_key()
if not API_KEY:
    st.error("TOGETHER_API_KEY missing.")
    st.stop()

@st.cache_resource(show_spinner=False)
def _client(key: str):
    return Together(api_key=key)

client = _client(API_KEY)

def _chat(messages, model: str, max_tokens: int = 280, temperature: float = 0.7, top_p: float = 0.9):
    def _call(m):
        r = client.chat.completions.create(
            model=m, messages=messages,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=False
        )
        return r.choices[0].message.content.strip()

    for attempt in range(5):
        try:
            return _call(model)
        except Exception as e:
            s = str(e)
            # fallback once to Free if Turbo is rate-limited
            if ("429" in s or "RateLimit" in s) and model.endswith("-Turbo"):
                try:
                    return _call(model + "-Free")
                except Exception:
                    pass
            if "429" in s or "RateLimit" in s or "temporarily unavailable" in s:
                time.sleep((2 ** attempt) + random.uniform(0, 0.5))
                continue
            raise
    raise RuntimeError("Rate limit retries exceeded")

def generate_reply(user_text: str, emotion: str, model: str):
    sys = (
        "You assist with journal reflections. Be concise, kind, and practical. "
        "Use the detected emotion as context; do not contradict it unless the user clarifies. "
        "Offer exactly one concrete step and one short quote (Bhagavad Gita, Bible, or Quran)."
    )
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Emotion: {emotion}\nJournal: {user_text}"},
    ]
    return _chat(msgs, model=model, max_tokens=280, temperature=0.7, top_p=0.9)

if "busy" not in st.session_state:
    st.session_state.busy = False

user_input = st.text_area("Write your entry", height=150, key="journal_text")

model_choice = st.selectbox(
    "Model",
    [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    ],
    index=0,
)

if st.button("Analyze & Respond") and not st.session_state.busy:
    if not user_input or not user_input.strip():
        st.warning("Please enter some text.")
    else:
        st.session_state.busy = True
        try:
            emo, conf, all_scores = detect_emotion_9(user_input)
            reply = generate_reply(user_input, emo, model_choice)

            st.markdown(f"Detected emotion (9-class): `{emo}` ({conf:.2f})")
            st.write(reply)

            if st.checkbox("Save this entry"):
                row = pd.DataFrame([[user_input, emo, conf, reply]], columns=["Journal", "Emotion9", "Confidence", "Response"])
                try:
                    exists = os.path.exists("journal_log.csv")
                    row.to_csv("journal_log.csv", mode="a", header=not exists, index=False)
                    st.info("Saved to journal_log.csv")
                    # clear input after save (no rerun needed)
                    st.session_state.journal_text = ""
                except Exception as e:
                    st.error(f"Could not save: {e}")

            if st.checkbox("Show emotion scores"):
                df_scores = (pd.DataFrame.from_dict(all_scores, orient="index", columns=["score"])
                             .sort_values("score", ascending=False))
                st.dataframe(df_scores.head(3))
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            st.session_state.busy = False
