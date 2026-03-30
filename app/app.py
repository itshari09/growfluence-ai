import streamlit as st
import joblib
import pandas as pd
import os
from textblob import TextBlob

import time

# SESSION STATE
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

# INTRO SCREEN
if not st.session_state.intro_done:
    st.markdown("""
    <style>
    .intro {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: black;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        z-index: 9999;
        animation: fadeOut 3s ease-in-out forwards;
    }

    .intro h1 {
        color: white;
        font-size: 50px;
        animation: zoomIn 2s ease;
    }

    @keyframes zoomIn {
        from {transform: scale(0.8); opacity: 0;}
        to {transform: scale(1); opacity: 1;}
    }

    @keyframes fadeOut {
        0% {opacity: 1;}
        80% {opacity: 1;}
        100% {opacity: 0; visibility: hidden;}
    }
    </style>

    <div class="intro">
        <h1>🚀 Growfluence AI</h1>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(3)
    st.session_state.intro_done = True
    st.rerun()

st.markdown("""
<style>
html {
    scroll-behavior: smooth;
}

.section {
    padding: 60px;
    text-align: center;
    color: white;
}

.hero {
    background: linear-gradient(135deg, #141e30, #243b55);
    border-radius: 20px;
    padding: 80px;
    margin-bottom: 20px;
    animation: fadeIn 1.5s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🔥 AI Influencer Growth Suite</h1>
    <p>Predict your content performance before posting</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Growfluence AI",
    page_icon="🚀",
    layout="wide"
)

# -------------------------------
# PATH SETUP (IMPORTANT)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'viral_model.pkl')

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# -------------------------------
# CAPTION ANALYSIS FUNCTION
# -------------------------------
def analyze_caption(text):
    blob = TextBlob(text)

    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        mood = "Positive 😊"
    elif sentiment < 0:
        mood = "Negative 😡"
    else:
        mood = "Neutral 😐"

    length = len(text.split())

    return sentiment, mood, length

# -------------------------------
# UI DESIGN
# -------------------------------
st.title("🚀 Growfluence AI")
st.markdown("### AI Influencer Growth Suite")
st.write("Predict and optimize your content before posting!")

st.divider()

# INPUT SECTION
st.subheader("📥 Enter Content Details")

caption = st.text_area("✍️ Caption", placeholder="Write your caption here...")

col1, col2 = st.columns(2)

with col1:
    comments = st.number_input("💬 Expected Comments", min_value=0)

with col2:
    shares = st.number_input("🔁 Expected Shares", min_value=0)

st.divider()

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("🚀 Analyze & Predict"):

    if model is None:
        st.error("⚠️ Model not found. Please train the model first.")
    elif caption.strip() == "":
        st.warning("⚠️ Please enter a caption!")
    else:
        try:
            # Prepare input
            input_data = pd.DataFrame({
                'caption': [caption],
                'comments': [comments],
                'shares': [shares]
            })

            # Prediction
            prediction = model.predict(input_data)
            likes = int(prediction[0])

            # Viral Score Calculation
            viral_score = min(100, (likes / 1000) * 100)

            # Caption Analysis
            sentiment, mood, length = analyze_caption(caption)

            # -------------------------------
            # OUTPUT SECTION
            # -------------------------------
            st.subheader("📊 Results")

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"🔥 Predicted Likes: {likes}")
                st.metric("📊 Viral Score", f"{viral_score:.2f}/100")
                st.progress(int(viral_score))

            with col2:
                st.info("🧠 Caption Insights")
                st.write(f"**Mood:** {mood}")
                st.write(f"**Length:** {length} words")
                st.write(f"**Sentiment Score:** {sentiment:.2f}")

            # Suggestions
            st.subheader("💡 Suggestions")

            if length < 5:
                st.warning("👉 Caption is too short. Add more detail.")
            elif length > 30:
                st.warning("👉 Caption is too long. Try shortening it.")

            if sentiment < 0:
                st.warning("👉 Negative tone may reduce engagement.")
            elif sentiment > 0.5:
                st.success("👉 Positive tone boosts engagement!")

            if comments < 50:
                st.info("👉 Try to encourage more comments (ask questions).")

            if shares < 20:
                st.info("👉 Add share-worthy content (tips, value, insights).")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.markdown("💡 Built with AI for smarter content growth")
