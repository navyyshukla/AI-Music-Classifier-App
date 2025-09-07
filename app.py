# app.py - UPDATED FOR CNN MODEL

import streamlit as st
import librosa
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import os
from datetime import datetime
import warnings
# --- NEW IMPORTS FOR CNN ---
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Page configuration (Your custom config is preserved)
st.set_page_config(
    page_title="AI Music Genre Classifier", 
    page_icon="🎵", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Your Custom CSS is completely preserved
st.markdown("""
<style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #ee5a52, #ff8e53, #ff6b9d);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50, #34495e);
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: linear-gradient(90deg, #56ab2f, #a8e6cf);
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #ff6b6b, #ee5a52);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- CHANGED CODE: Load the new CNN model and class names ---
@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model('best_cnn_model.keras')
        class_names = np.load('class_names.npy', allow_pickle=True)
        return model, class_names, True
    except (IOError, OSError) as e:
        return None, None, False

model, class_names, MODEL_LOADED = load_models()

# --- NEW FUNCTION: Preprocesses a single audio file into an image tensor ---
def preprocess_audio_for_cnn(audio_path, duration=30, n_mels=128, fmax=8000, target_size=(128, 431)):
    """
    Converts an audio file into a preprocessed image tensor that matches
    the input requirements of the trained CNN model.
    """
    try:
        # 1. Load Audio
        y, sr = librosa.load(audio_path, duration=duration)
        if len(y) < duration * sr: # Pad short audio clips
            y = np.pad(y, (0, duration * sr - len(y)))
            
        # 2. Create Mel Spectrogram (using same parameters as training)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # 3. Convert spectrogram to a clean image in memory
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=fmax, ax=ax)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        # 4. Load image with PIL and preprocess for the model
        img = Image.open(buf).convert('RGB')
        img = img.resize((target_size[1], target_size[0])) # Note: PIL resize is (width, height)
        
        # 5. Convert to tensor and add a batch dimension
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        return img_array, y, sr
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None, None, None

# --- UNCHANGED: Your visualization functions are preserved ---
def create_feature_plots(audio_data, sr):
    fig = make_subplots(rows=1, cols=1, subplot_titles=["Audio Waveform"])
    time_axis = np.linspace(0, len(audio_data)/sr, len(audio_data))
    fig.add_trace(go.Scatter(x=time_axis, y=audio_data, mode='lines', name='Waveform', line=dict(color='#ff6b6b')), row=1, col=1)
    fig.update_layout(height=400, showlegend=False, title_text="Audio Waveform Analysis", title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_confidence_chart(probabilities, genre_names):
    fig = go.Figure(data=[go.Bar(x=genre_names, y=probabilities * 100, marker=dict(color=probabilities, colorscale='Viridis', showscale=True, colorbar=dict(title="Confidence %")), text=[f"{prob*100:.1f}%" for prob in probabilities], textposition='auto')])
    fig.update_layout(title="Genre Confidence Scores", xaxis_title="Music Genres", yaxis_title="Confidence (%)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=400)
    return fig

# --- UNCHANGED: Your custom header and sidebar ---
st.markdown("""<div class="main-header"><h1 style="color: white; margin: 0; font-size: 3rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🎵 AI Music Genre Classifier</h1><p style="color: white; margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">Powered by Deep Learning & Advanced Audio Signal Processing</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Advanced Settings")
    st.markdown("### 🎚️ Audio Processing")
    duration = st.slider("Analysis Duration (seconds)", 10, 60, 30)
    # The n_mfcc slider is no longer relevant for the CNN model, so it has been removed.
    st.markdown("### 🔍 Feature Analysis")
    show_waveform = st.checkbox("Show Audio Visualizations", value=True)
    show_confidence = st.checkbox("Show Confidence Breakdown", value=True)
    
    if MODEL_LOADED:
        st.markdown("### 📊 Model Info")
        st.success("✅ Model Loaded Successfully")
        st.write(f"**Genres:** {len(class_names)}")
        with st.expander("View All Genres"):
            for i, genre in enumerate(class_names):
                st.write(f"{i+1}. {genre.capitalize()}")
    else:
        st.markdown("### ❌ Model Status")
        st.error("Model files missing!")
        st.markdown("""**Required files:**\n- `best_cnn_model.keras`\n- `class_names.npy`""")

# Main content area (Your layout is preserved)
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📁 Upload Your Music")
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "flac", "ogg"], help="Supported formats: WAV, MP3, FLAC, OGG. Optimal length: 30 seconds")
    
    if uploaded_file is not None:
        file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB", "Upload time": datetime.now().strftime("%H:%M:%S")}
        st.markdown("### 📋 File Information")
        info_cols = st.columns(3)
        for i, (key, value) in enumerate(file_details.items()):
            with info_cols[i]:
                st.metric(label=key, value=value)
        st.markdown("### 🎧 Audio Player")
        st.audio(uploaded_file, format='audio/wav')

with col2:
    st.markdown("## 📈 Quick Stats")
    if MODEL_LOADED:
        stats_data = {"Model Type": "CNN (TensorFlow)", "Features": "Mel Spectrogram Images", "Processing": "Real-time", "Accuracy": "High Precision"}
        for stat, value in stats_data.items():
            st.markdown(f"""<div class="metric-card"><h4 style="margin:0; color: white;">{stat}</h4><p style="margin:0; color: #a0a0a0;">{value}</p></div>""", unsafe_allow_html=True)

if uploaded_file is not None and MODEL_LOADED:
    st.markdown("---")
    st.markdown("## 🎯 Genre Classification")
    
    classify_col1, classify_col2, classify_col3 = st.columns([1, 2, 1])
    with classify_col2:
        if st.button("🚀 Analyze Music Genre", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📁 Saving audio file...")
            progress_bar.progress(10)
            file_extension = uploaded_file.name.split('.')[-1].lower()
            temp_filename = f"temp.{file_extension}"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # --- CHANGED CODE: New prediction logic for CNN model ---
            status_text.text("🖼️ Converting audio to spectrogram...")
            progress_bar.progress(30)
            time.sleep(0.5)
            
            image_tensor, audio_data, sr = preprocess_audio_for_cnn(temp_filename, duration=duration)
            
            if image_tensor is not None:
                status_text.text("🧠 Making prediction...")
                progress_bar.progress(80)
                time.sleep(0.3)
                
                prediction_probabilities = model.predict(image_tensor)[0]
                prediction_index = np.argmax(prediction_probabilities)
                predicted_genre = class_names[prediction_index]
                confidence = np.max(prediction_probabilities) * 100
                
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # --- UNCHANGED: Your beautiful results display is preserved ---
                result_col1, result_col2 = st.columns([1, 1])
                with result_col1:
                    st.markdown(f"""<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 20px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin: 1rem 0;"><h2 style="color: white; margin: 0;">🎵 Predicted Genre</h2><h1 style="color: #ffff00; margin: 10px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">{predicted_genre.upper()}</h1><h3 style="color: white; margin: 0;">Confidence: {confidence:.1f}%</h3></div>""", unsafe_allow_html=True)
                
                with result_col2:
                    if confidence >= 80: confidence_color, confidence_text = "#2ecc71", "Very High"
                    elif confidence >= 60: confidence_color, confidence_text = "#f39c12", "High"
                    elif confidence >= 40: confidence_color, confidence_text = "#e67e22", "Medium"
                    else: confidence_color, confidence_text = "#e74c3c", "Low"
                    st.markdown(f"""<div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 20px; text-align: center; margin: 1rem 0;"><h3 style="color: white;">Confidence Level</h3><div style="background: #333; border-radius: 25px; padding: 5px; margin: 1rem 0;"><div style="background: {confidence_color}; width: {confidence}%; height: 30px; border-radius: 20px; display: flex; align-items: center; justify-content: center;"><span style="color: white; font-weight: bold;">{confidence:.1f}%</span></div></div><p style="color: {confidence_color}; font-size: 1.2rem; margin: 0;">{confidence_text}</p></div>""", unsafe_allow_html=True)
                
                if show_confidence:
                    st.markdown("### 📊 Genre Confidence Breakdown")
                    confidence_fig = create_confidence_chart(prediction_probabilities, class_names)
                    st.plotly_chart(confidence_fig, use_container_width=True)
                    top_3_indices = np.argsort(prediction_probabilities)[-3:][::-1]
                    st.markdown("#### 🏆 Top 3 Predictions")
                    top3_cols = st.columns(3)
                    medals = ["🥇", "🥈", "🥉"]
                    colors = ["#ffd700", "#c0c0c0", "#cd7f32"]
                    for i, idx in enumerate(top_3_indices):
                        genre, prob = class_names[idx], prediction_probabilities[idx] * 100
                        with top3_cols[i]:
                            st.markdown(f"""<div style="background: {colors[i]}; background: linear-gradient(135deg, {colors[i]}, {colors[i]}88); padding: 1rem; border-radius: 15px; text-align: center; margin: 0.5rem 0;"><h3 style="margin: 0; color: white;">{medals[i]} {genre.capitalize()}</h3><p style="margin: 0; color: white; font-size: 1.1rem;">{prob:.1f}%</p></div>""", unsafe_allow_html=True)
                
                if show_waveform and audio_data is not None:
                    st.markdown("### 🌊 Audio Analysis Visualization")
                    viz_fig = create_feature_plots(audio_data, sr)
                    st.plotly_chart(viz_fig, use_container_width=True)
                
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            else:
                st.error("❌ Failed to process the audio file.")

# --- UNCHANGED: Your footer and welcome screen are preserved ---
if not uploaded_file:
    st.markdown("---")
    st.markdown("## 🚀 How It Works")
    info_cols = st.columns(4)
    steps = [("📤", "Upload", "Upload your audio file"), ("🖼️", "Analyze", "Convert audio to a spectrogram image"), ("🧠", "Classify", "CNN model predicts the genre"), ("📊", "Results", "View detailed analysis scores")]
    for i, (icon, title, desc) in enumerate(steps):
        with info_cols[i]:
            st.markdown(f"""<div style="text-align: center; padding: 1.5rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin: 0.5rem 0;"><div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div><h4 style="color: white; margin: 0.5rem 0;">{title}</h4><p style="color: #a0a0a0; font-size: 0.9rem; margin: 0;">{desc}</p></div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""<div style="text-align: center; padding: 2rem; color: #a0a0a0;"><p>🎵 Built with Streamlit, Librosa, and TensorFlow ✨</p><p>Upload your music and discover its genre with AI-powered analysis!</p></div>""", unsafe_allow_html=True)