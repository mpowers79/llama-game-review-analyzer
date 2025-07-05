# File: review_app.py
# Description: Streamlit app to manually run game review processor to extract sentiment and keywords from game reviews
#   Normally you might want to run the process_reviews.py file from chron job or similar
#
# Copyright (c) 2025 Michael Powers
#
# Usage:
#   streamlit run review_app.py
# 
#


import streamlit as st
from process_reviews import process_reviews
import os
from streamlit_autorefresh import st_autorefresh


def get_file_contents(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    return None


def get_current_status():
    status = get_file_contents("status.log")
    if status is None:
        return "Processor is not running..."
    return status


count = st_autorefresh(interval=5000, limit=1000, key="refreshcounter")

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Game Review Processor")

st.title("🎮 Game Review Processor")

col1, col2, col3, col4 = st.columns(4, vertical_alignment="center")

with col1:
    if st.button("Run"):
        if 'max_reviews' in st.session_state.keys():
            reviews = int(st.session_state.max_reviews)
        else:
            reviews = 20
        process_reviews(reviews)

with col2:
    st.session_state.max_reviews = st.text_input("Max Reviews to Process", value = "20")

st.markdown(f"**Processor Status:** {get_file_contents('process.log')}")

#ideally: show a progress bar while running
#if not running, show last run date/time
#we can store this in a status file
#----- Progress bar --------
progress = get_file_contents("progress.log")
if progress is not None:
    val = float(progress)
    my_bar = st.progress(val, text="Processing reviews. Please wait.")

#----- Status Text --------
st.code(get_current_status(), language='text')

