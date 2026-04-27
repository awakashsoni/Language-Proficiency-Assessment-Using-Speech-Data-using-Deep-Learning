import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
import tempfile
import plotly.graph_objects as go

# CONFIG
st.set_page_config(page_title="AISpeech", layout="wide")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# STYLE
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.center-title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    margin-bottom: 20px;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
}
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border-radius: 12px;
    height: 55px;
    font-size: 18px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

#  TITLE 
st.markdown('<div class="center-title">🎤 AI Speech Dashboard</div>', unsafe_allow_html=True)

#  MODEL 
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()
classes = ["Beginner", "Intermediate", "Advanced"]

#  FEATURE
def extract_features(file):
    audio, sr = librosa.load(file, sr=16000, duration=5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# INPUT
col1, col2 = st.columns(2)

file_to_process = None

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📂 Upload File")
    uploaded_file = st.file_uploader("", type=["wav","mp3"])
    if uploaded_file:
        file_to_process = uploaded_file
        st.audio(uploaded_file)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎤 Record Voice")
    audio_bytes = st.audio_input("Record Speech")
    if audio_bytes:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(audio_bytes.read())
        file_to_process = temp_file.name
        st.audio(audio_bytes)
    st.markdown('</div>', unsafe_allow_html=True)

# ANALYZE BUTTON
colA, colB, colC = st.columns([2,1,2])

with colB:
    analyze = st.button("🚀 Analyze")

#PROCESS
if file_to_process and analyze:

    features = extract_features(file_to_process).reshape(1,40)
    prediction = model.predict(features)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown("## 📊 Results Panel")

    r1, r2 = st.columns(2)

    #  GAUGE
    with r1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00ff99"},
                'steps': [
                    {'range': [0, 40], 'color': "#ff4d4d"},
                    {'range': [40, 70], 'color': "#ffc107"},
                    {'range': [70, 100], 'color': "#00ff99"},
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    #BAR CHART 
    with r2:
        fig2 = go.Figure(data=[
            go.Bar(
                x=classes,
                y=prediction[0]*100,
                marker_color=["#ff4d4d","#ffc107","#00ff99"]
            )
        ])
        fig2.update_layout(title="Class Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    #  RESULT
    st.success(f"🎯 Prediction: {predicted_class}")

    #  DETAILS 
    st.markdown("### 📋 Detailed Scores")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {prediction[0][i]*100:.2f}%")