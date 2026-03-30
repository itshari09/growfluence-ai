import streamlit as st
import joblib
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.text_analysis import analyze_caption


st.set_page_config(page_title="Growfluence AI", layout="wide")

st.title("🚀 Growfluence AI")
st.markdown("### AI Influencer Growth Suite")

st.divider()

# Inputs
caption = st.text_area("✍️ Enter your caption")

col1, col2 = st.columns(2)

with col1:
    comments = st.number_input("💬 Comments", min_value=0)

with col2:
    shares = st.number_input("🔁 Shares", min_value=0)

if st.button("🚀 Analyze & Predict"):

    try:
        model = joblib.load('../models/viral_model.pkl')

        input_data = pd.DataFrame({
            'caption': [caption],
            'comments': [comments],
            'shares': [shares]
        })

        prediction = model.predict(input_data)
        likes = int(prediction[0])
        viral_score = min(100, (likes / 1000) * 100)

        # Caption analysis
        analysis = analyze_caption(caption)

        # OUTPUT UI
        st.success(f"🔥 Predicted Likes: {likes}")
        st.progress(int(viral_score))
        st.write(f"📊 Viral Score: {viral_score:.2f}/100")

        st.subheader("🧠 Caption Analysis")
        st.write(f"Mood: {analysis['mood']}")
        st.write(f"Length: {analysis['length']} words")

    except:
        st.error("⚠️ Train model first!")