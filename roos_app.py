
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="ROOS: Resonant Operating System", layout="wide")

# In-memory log storage
if "inbot_memory" not in st.session_state:
    st.session_state.inbot_memory = ["this is a test", "another symbolic message"]
    st.session_state.outbot_memory = ["this is a test", "another symbolic message", "extra"]
    st.session_state.similarity_scores = []

# Title
st.title("ROOS: Resonant Operating System (Minimal Prototype)")

# Hypothesis Input
st.subheader("Submit a Hypothesis")
hypo = st.text_input("What do you suspect about the system, cosmos, or self?")
if st.button("Submit Hypothesis"):
    st.success("Hypothesis logged: " + hypo)

# Bot Divergence Visual
st.subheader("Memory Divergence")
in_lens = [len(m) for m in st.session_state.inbot_memory]
out_lens = [len(m) for m in st.session_state.outbot_memory]
st.line_chart({"inbot": in_lens, "outbot": out_lens})

# Divergence Similarity
st.subheader("Cosmic Divergence Metric")
def evaluate_divergence():
    memory = st.session_state.inbot_memory + st.session_state.outbot_memory
    tfidf = TfidfVectorizer().fit_transform(memory)
    score = cosine_similarity(tfidf).mean()
    st.session_state.similarity_scores.append(score)
    return score

if st.button("Evaluate"):
    score = evaluate_divergence()
    st.metric("Similarity Score", f"{score:.4f}")

# Trend Chart
if st.session_state.similarity_scores:
    st.line_chart(st.session_state.similarity_scores)

# File Upload (TXT only)
st.subheader("Upload File for Bitstream Analysis")
uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    bitstream = ''.join(format(ord(c), '08b') for c in content)
    st.text_area("Bitstream", bitstream[:512] + "..." if len(bitstream) > 512 else bitstream)
    st.session_state.inbot_memory.append(content)
    st.success("Content ingested into inbot memory.")
