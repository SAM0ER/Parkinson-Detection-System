import streamlit as st
import numpy as np
import pandas as pd
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
from streamlit_mic_recorder import mic_recorder
matplotlib.use('Agg')
from pydub import AudioSegment
import io
import os
import tempfile
from fpdf import FPDF
import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import parselmouth
    from parselmouth.praat import call
    PRAAT_AVAILABLE = True
except ImportError:
    PRAAT_AVAILABLE = False

# ══════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════
#  CSS STYLING
# ══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');
* { font-family: 'Cairo', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }

.hero-box {
    background: linear-gradient(135deg, #1e3a5f, #0d2137);
    border: 1px solid #2a5298;
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(42, 82, 152, 0.3);
}
.hero-title { font-size: 2.8rem; font-weight: 700; color: #ffffff; margin: 0; }
.hero-subtitle { font-size: 1.1rem; color: #90b4e8; margin-top: 10px; }

.result-positive {
    background: linear-gradient(135deg, #3d0000, #6b0000);
    border: 2px solid #ff4444;
    border-radius: 15px; padding: 30px; text-align: center; color: white;
}
.result-negative {
    background: linear-gradient(135deg, #003d1f, #005a2c);
    border: 2px solid #00cc66;
    border-radius: 15px; padding: 30px; text-align: center; color: white;
}
.result-emoji { font-size: 4rem; }
.result-title { font-size: 1.8rem; font-weight: 700; margin: 10px 0; }

.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 20px; text-align: center; color: white;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #4da6ff; }
.metric-label { font-size: 0.9rem; color: #aaa; }

.feature-info {
    background: rgba(42,82,152,0.15);
    border-left: 3px solid #2a5298;
    border-radius: 8px; padding: 15px;
    color: #ccc; font-size: 0.9rem; margin-bottom: 10px;
}
.section-title {
    color: #4da6ff; font-size: 1.3rem; font-weight: 700;
    border-bottom: 2px solid #2a5298;
    padding-bottom: 8px; margin-bottom: 20px;
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a, #1a2740);
    border-right: 1px solid #2a3f5f;
}
.stButton > button {
    background: linear-gradient(135deg, #2a5298, #1a3a6e);
    color: white; border: none; border-radius: 10px;
    padding: 12px 30px; font-size: 1rem; font-weight: 600;
    width: 100%; transition: all 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3a6bc4, #2a5298);
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(42,82,152,0.5);
}
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #90b4e8 !important; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  LOAD MODELS  (Extra Trees = best, Random Forest)
# ══════════════════════════════════════════════════
@st.cache_resource
def load_models():
    try:
        best_model = joblib.load("best_model.pkl")          # Extra Trees pipeline
        rf_model   = joblib.load("random_forest_model.pkl") # Random Forest pipeline
        return best_model, rf_model, True
    except Exception as e:
        return None, None, False

best_model, rf_model, models_loaded = load_models()


# ══════════════════════════════════════════════════
#  FEATURE NAMES & DESCRIPTIONS
# ══════════════════════════════════════════════════
FEATURE_NAMES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

FEATURE_DESCRIPTIONS = {
    'MDVP:Fo(Hz)':       'Average vocal fundamental frequency',
    'MDVP:Fhi(Hz)':      'Maximum vocal fundamental frequency',
    'MDVP:Flo(Hz)':      'Minimum vocal fundamental frequency',
    'MDVP:Jitter(%)':    'Frequency variation percentage',
    'MDVP:Jitter(Abs)':  'Absolute frequency variation',
    'MDVP:RAP':          'Relative amplitude perturbation',
    'MDVP:PPQ':          'Five-point period perturbation quotient',
    'Jitter:DDP':        'Average absolute difference of periods',
    'MDVP:Shimmer':      'Amplitude variation',
    'MDVP:Shimmer(dB)':  'Amplitude variation in dB',
    'Shimmer:APQ3':      '3-point amplitude perturbation quotient',
    'Shimmer:APQ5':      '5-point amplitude perturbation quotient',
    'MDVP:APQ':          '11-point amplitude perturbation quotient',
    'Shimmer:DDA':       'Average absolute differences between cycles',
    'NHR':               'Noise-to-harmonic ratio',
    'HNR':               'Harmonic-to-noise ratio',
    'RPDE':              'Recurrence period density entropy',
    'DFA':               'Detrended fluctuation analysis',
    'spread1':           'Nonlinear fundamental frequency measure 1',
    'spread2':           'Nonlinear fundamental frequency measure 2',
    'D2':                'Correlation dimension',
    'PPE':               'Pitch period entropy'
}


# ══════════════════════════════════════════════════
#  FEATURE EXTRACTION — PRAAT
# ══════════════════════════════════════════════════
def extract_features_praat(audio_path):
    try:
        sound = parselmouth.Sound(audio_path)
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        fo  = call(pitch, "Get mean",    0, 0, "Hertz")
        fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

        if fo <= 0 or fhi <= 0 or flo <= 0:
            return None, "Could not detect voice pitch. Please speak clearly and try again."

        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter_pct = call(point_process, "Get jitter (local)",           0, 0, 0.0001, 0.02, 1.3)
        jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rap        = call(point_process, "Get jitter (rap)",             0, 0, 0.0001, 0.02, 1.3)
        ppq        = call(point_process, "Get jitter (ppq5)",            0, 0, 0.0001, 0.02, 1.3)
        ddp        = call(point_process, "Get jitter (ddp)",             0, 0, 0.0001, 0.02, 1.3)

        shimmer    = call([sound, point_process], "Get shimmer (local)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3       = call([sound, point_process], "Get shimmer (apq3)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5       = call([sound, point_process], "Get shimmer (apq5)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
        mdvp_apq   = call([sound, point_process], "Get shimmer (apq11)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
        dda        = call([sound, point_process], "Get shimmer (dda)",      0, 0, 0.0001, 0.02, 1.3, 1.6)

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        nhr = 1.0 / hnr if hnr > 0 else 0.5

        y, sr = librosa.load(audio_path, sr=None)
        mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        rpde    = float(np.clip(np.std(mfcc[0]) / (np.mean(np.abs(mfcc[0])) + 1e-6), 0.25, 0.85))
        zcr     = librosa.feature.zero_crossing_rate(y)[0]
        dfa     = float(np.clip(np.mean(zcr) * 10, 0.5, 0.9))
        sc      = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spread1 = float(-np.clip(np.std(sc) / 100, 3, 8))
        spread2 = float( np.clip(np.mean(sc) / 5000, 0.1, 0.5))
        chroma  = librosa.feature.chroma_stft(y=y, sr=sr)
        d2      = float(np.clip(np.std(chroma), 1.5, 3.5))
        f0_lib, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=600)
        f0_vals = f0_lib[voiced_flag] if voiced_flag is not None and np.any(voiced_flag) else np.array([fo])
        ppe     = float(np.clip(np.std(f0_vals) / np.mean(f0_vals) if len(f0_vals) > 1 else 0.2, 0.05, 0.6))

        features = np.array([
            fo, fhi, flo,
            jitter_pct * 100, jitter_abs,
            rap, ppq, ddp,
            shimmer, shimmer_db,
            abs(apq3), abs(apq5), abs(mdvp_apq), abs(dda),
            abs(nhr), hnr,
            rpde, dfa, spread1, spread2, d2, ppe
        ], dtype=np.float64)
        return features, None
    except Exception as e:
        return None, f"Praat error: {str(e)}"


# ══════════════════════════════════════════════════
#  FEATURE EXTRACTION — LIBROSA (fallback)
# ══════════════════════════════════════════════════
def extract_features_librosa(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_values = f0[voiced_flag]

        if len(f0_values) < 5:
            return None, "Audio too short or no clear voice detected."

        fo  = np.mean(f0_values)
        fhi = np.max(f0_values)
        flo = np.min(f0_values)

        periods    = 1.0 / f0_values
        diffs      = np.abs(np.diff(periods))
        avg_period = np.mean(periods)

        jitter_pct = (np.mean(diffs) / avg_period) * 100
        jitter_abs = np.mean(diffs) * 1000
        rap = np.mean([np.abs(periods[i] - np.mean(periods[max(0,i-1):i+2]))
                       for i in range(1, len(periods)-1)]) / avg_period
        ppq = np.mean([np.abs(periods[i] - np.mean(periods[max(0,i-2):i+3]))
                       for i in range(2, len(periods)-2)]) / avg_period
        ddp = 3 * rap

        rms       = librosa.feature.rms(y=y)[0]
        rms_vals  = rms[rms > 0] if np.any(rms > 0) else rms + 1e-10
        amp_diffs = np.abs(np.diff(rms_vals))
        avg_amp   = np.mean(rms_vals)
        shimmer    = np.mean(amp_diffs) / avg_amp if avg_amp > 0 else 0
        shimmer_db = 20 * np.log10(1 + shimmer) if shimmer > 0 else 0
        apq3       = shimmer * 0.7
        apq5       = shimmer * 0.85
        mdvp_apq   = shimmer * 0.95
        dda        = 3 * apq3

        stft = np.abs(librosa.stft(y))
        harmonic, percussive = librosa.decompose.hpss(stft)
        h_energy = np.mean(harmonic**2)
        p_energy = np.mean(percussive**2)
        hnr = 10 * np.log10(h_energy / (p_energy + 1e-10) + 1e-10)
        nhr = 1.0 / (hnr + 1e-6) if hnr > 0 else 0.5

        mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        rpde    = float(np.clip(np.std(mfcc[0]) / (np.mean(np.abs(mfcc[0])) + 1e-6), 0.25, 0.85))
        zcr     = librosa.feature.zero_crossing_rate(y)[0]
        dfa     = float(np.clip(np.mean(zcr) * 10, 0.5, 0.9))
        sc      = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spread1 = float(-np.clip(np.std(sc) / 100, 3, 8))
        spread2 = float( np.clip(np.mean(sc) / 5000, 0.1, 0.5))
        chroma  = librosa.feature.chroma_stft(y=y, sr=sr)
        d2      = float(np.clip(np.std(chroma), 1.5, 3.5))
        ppe     = float(np.clip(np.std(f0_values) / fo, 0.05, 0.6))

        features = np.array([
            fo, fhi, flo,
            jitter_pct, jitter_abs, rap, ppq, ddp,
            shimmer, shimmer_db, abs(apq3), abs(apq5), abs(mdvp_apq), abs(dda),
            abs(nhr), hnr,
            rpde, dfa, spread1, spread2, d2, ppe
        ], dtype=np.float64)
        return features, None
    except Exception as e:
        return None, f"Error processing file: {str(e)}"


def extract_features_from_audio(audio_path):
    if PRAAT_AVAILABLE:
        features, error = extract_features_praat(audio_path)
        if features is not None:
            return features, None
    return extract_features_librosa(audio_path)


# ══════════════════════════════════════════════════
#  WAVEFORM PLOT
# ══════════════════════════════════════════════════
def plot_waveform(audio_path):
    y, sr    = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    time     = np.linspace(0, duration, len(y))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.patch.set_facecolor('#1a1f2e')

    axes[0].plot(time, y, color='#4da6ff', linewidth=0.5, alpha=0.8)
    axes[0].set_facecolor('#0d1b2a')
    axes[0].set_xlabel('Time (seconds)', color='white')
    axes[0].set_ylabel('Amplitude', color='white')
    axes[0].set_title('Audio Waveform', color='white', fontsize=14, pad=15)
    axes[0].tick_params(colors='white')
    for spine in axes[0].spines.values(): spine.set_edgecolor('#2a5298')

    D   = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz',
                                   ax=axes[1], cmap='plasma')
    axes[1].set_facecolor('#0d1b2a')
    axes[1].set_title('Spectrogram', color='white', fontsize=14, pad=15)
    axes[1].set_xlabel('Time (seconds)', color='white')
    axes[1].set_ylabel('Frequency (Hz)', color='white')
    axes[1].tick_params(colors='white')
    for spine in axes[1].spines.values(): spine.set_edgecolor('#2a5298')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════
#  PREDICT  — Extra Trees (best) + Random Forest
#  Both pipelines already include scaler + SMOTE
# ══════════════════════════════════════════════════
def predict(features):
    if not models_loaded:
        return None

    features_2d = features.reshape(1, -1)
    results = {}

    # ── Extra Trees ──────────────────────────────
    try:
        et_pred  = best_model.predict(features_2d)[0]
        et_proba = best_model.predict_proba(features_2d)[0]
        results['Extra Trees'] = {
            'prediction':     int(et_pred),
            'confidence':     float(max(et_proba)) * 100,
            'parkinson_prob': float(et_proba[1]) * 100
        }
    except Exception as e:
        results['Extra Trees'] = {'error': str(e)}

    # ── Random Forest ────────────────────────────
    try:
        rf_pred  = rf_model.predict(features_2d)[0]
        rf_proba = rf_model.predict_proba(features_2d)[0]
        results['Random Forest'] = {
            'prediction':     int(rf_pred),
            'confidence':     float(max(rf_proba)) * 100,
            'parkinson_prob': float(rf_proba[1]) * 100
        }
    except Exception as e:
        results['Random Forest'] = {'error': str(e)}

    return results


# ══════════════════════════════════════════════════
#  PDF REPORT
# ══════════════════════════════════════════════════
def generate_pdf_report(patient_name, features, results, method):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font('Arial', 'B', 20)
    pdf.set_fill_color(15, 25, 50)
    pdf.set_text_color(255, 255, 255)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.cell(0, 15, "Parkinson's Disease Detection Report", ln=True, align='C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, "AI-Powered Voice Analysis System", ln=True, align='C')

    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 10, "Patient Information", ln=True, fill=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Name: {patient_name if patient_name else 'Anonymous'}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 8, f"Analysis Method: {method}", ln=True)
    engine = "Praat (parselmouth)" if PRAAT_AVAILABLE else "librosa"
    pdf.cell(0, 8, f"Feature Extraction Engine: {engine}", ln=True)
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 10, "Prediction Results", ln=True, fill=True)

    for model_name, res in results.items():
        if 'error' not in res:
            status = "POSITIVE - Parkinson's Detected" if res['prediction'] == 1 else "NEGATIVE - Healthy"
            pdf.set_text_color(200, 0, 0) if res['prediction'] == 1 else pdf.set_text_color(0, 150, 0)
            pdf.set_font('Arial', 'B', 13)
            pdf.cell(0, 10, f"  {model_name}: {status}", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Arial', '', 11)
            pdf.cell(0, 8,
                f"  Confidence: {res['confidence']:.1f}%  |  Parkinson Probability: {res['parkinson_prob']:.1f}%",
                ln=True)
            pdf.ln(3)

    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 10, "Extracted Voice Features", ln=True, fill=True)
    pdf.set_font('Arial', '', 10)
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, features)):
        pdf.cell(95, 7, f"{name}: {val:.6f}", border=1)
        if i % 2 == 1:
            pdf.ln()
    if len(features) % 2 != 0:
        pdf.ln()

    pdf.ln(5)
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 6,
        "DISCLAIMER: This report is generated by an AI system for research purposes only. "
        "It is NOT a medical diagnosis. Please consult a qualified neurologist for clinical evaluation.")

    return bytes(pdf.output())


# ══════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0;'>
        <span style='font-size:3rem;'>🧠</span>
        <h2 style='color:#4da6ff; margin:10px 0 5px;'>Parkinson's AI</h2>
        <p style='color:#90b4e8; font-size:0.85rem;'>Detection System v2.0</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📊 Model Status")
    if models_loaded:
        st.success("✅ Extra Trees  — Loaded  (AUC 99.14%)")
        st.success("✅ Random Forest — Loaded  (AUC 96.55%)")
    else:
        st.error("❌ Failed to load models")
        st.info("Make sure these files exist:\n- best_model.pkl\n- random_forest_model.pkl")

    st.divider()
    st.markdown("### 🔬 Feature Engine")
    if PRAAT_AVAILABLE:
        st.success("✅ Praat (High Accuracy)")
    else:
        st.warning("⚠️ librosa (Basic)\n\nFor better accuracy:\n`pip install praat-parselmouth`")

    st.divider()
    st.markdown("### 🗂️ Select Mode")
    mode = st.radio(
        "",
        ["🎤 Upload Audio", "🎙️ Live Recording", "✍️ Manual Input", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()
    patient_name = st.text_input("👤 Patient Name (optional)", placeholder="Enter name...")

    st.markdown("""
    <div style='text-align:center; color:#555; font-size:0.75rem; margin-top:30px;'>
        Graduation Project 2026<br>
        Parkinson's Disease Detection<br>
        Using Machine Learning
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🧠 Parkinson's Disease Detection</div>
    <div class="hero-subtitle">
        AI System for Parkinson's Detection through Voice Analysis
        <br>
        <small>Extra Trees &amp; Random Forest | SMOTE | Praat Voice Analysis | Real-time Analysis</small>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  HELPER: show_results
# ══════════════════════════════════════════════════
def show_results(results, features, method):
    if not results:
        return

    st.markdown('<div class="section-title">🎯 Prediction Results</div>', unsafe_allow_html=True)

    for model_name, res in results.items():
        if 'error' not in res:
            is_positive = res['prediction'] == 1
            css_class   = "result-positive" if is_positive else "result-negative"
            emoji       = "⚠️" if is_positive else "✅"
            title       = "POSITIVE - Parkinson's Detected" if is_positive else "NEGATIVE - Healthy"
            color       = "#ff6666" if is_positive else "#66ff99"
            st.markdown(f"""
            <div class="{css_class}">
                <div class="result-emoji">{emoji}</div>
                <div class="result-title">{model_name}: {title}</div>
                <div style="font-size:1.1rem; color:{color};">
                    Confidence: {res['confidence']:.1f}% &nbsp;|&nbsp;
                    Parkinson Probability: {res['parkinson_prob']:.1f}%
                </div>
            </div><br>
            """, unsafe_allow_html=True)

    # ── Comparison chart ─────────────────────────
    st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)
    models_list     = [m for m in results if 'error' not in results[m]]
    healthy_probs   = [100 - results[m]['parkinson_prob'] for m in models_list]
    parkinson_probs = [results[m]['parkinson_prob']       for m in models_list]

    fig2, ax = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor('#1a1f2e')
    ax.set_facecolor('#0d1b2a')
    x     = np.arange(len(models_list))
    bars1 = ax.bar(x - 0.2, healthy_probs,   0.35, label='Healthy',     color='#00cc66', alpha=0.85)
    bars2 = ax.bar(x + 0.2, parkinson_probs, 0.35, label="Parkinson's", color='#ff4444', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, color='white', fontsize=13)
    ax.set_ylabel('Probability %', color='white')
    ax.set_ylim(0, 115)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1f2e', labelcolor='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#2a5298')
    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{bar.get_height():.1f}%', ha='center', color='white', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{bar.get_height():.1f}%', ha='center', color='white', fontsize=10)
    st.pyplot(fig2)
    plt.close()

    # ── Features table ───────────────────────────
    with st.expander("🔬 View Extracted Features"):
        df_feat = pd.DataFrame({
            'Feature':     FEATURE_NAMES,
            'Value':       [f"{v:.6f}" for v in features],
            'Description': [FEATURE_DESCRIPTIONS[f] for f in FEATURE_NAMES]
        })
        st.dataframe(df_feat, use_container_width=True, height=400)

    # ── PDF download ─────────────────────────────
    st.markdown('<div class="section-title">📄 Download Report</div>', unsafe_allow_html=True)
    pdf_bytes = generate_pdf_report(patient_name, features, results, method)
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_bytes,
        file_name=f"parkinson_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )


# ══════════════════════════════════════════════════
#  MODE: UPLOAD AUDIO
# ══════════════════════════════════════════════════
if mode == "🎤 Upload Audio":
    st.markdown('<div class="section-title">🎤 Upload Audio File for Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload WAV or MP3 file",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Recommended: record a sustained 'Aah' vowel for 3-5 seconds"
        )
        engine_label = "Praat (High Accuracy)" if PRAAT_AVAILABLE else "librosa (Basic)"
        st.markdown(f"""
        <div class="feature-info">
        💡 <b>Recording tip:</b><br>
        Record the vowel "Aah" for 3-5 seconds in a steady, continuous voice.<br><br>
        🔬 <b>Feature Engine:</b> {engine_label}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">22</div>
            <div class="metric-label">Features extracted automatically</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">2</div>
            <div class="metric-label">AI models for comparison</div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        st.audio(uploaded_file)

        if st.button("🔍 Analyze & Predict", type="primary"):
            with st.spinner("⏳ Extracting voice features..."):
                features, error = extract_features_from_audio(tmp_path)

            if error:
                st.error(f"❌ {error}")
            else:
                st.markdown('<div class="section-title">📈 Voice Signal Analysis</div>',
                            unsafe_allow_html=True)
                with st.spinner("Plotting waveform..."):
                    fig = plot_waveform(tmp_path)
                    st.pyplot(fig)
                    plt.close()
                with st.spinner("🤖 Running prediction models..."):
                    results = predict(features)
                show_results(results, features, "Audio File Upload")

        os.unlink(tmp_path)


# ══════════════════════════════════════════════════
#  MODE: LIVE RECORDING
# ══════════════════════════════════════════════════
elif mode == "🎙️ Live Recording":

    st.markdown(
        '<div class="section-title">🎙️ Live Microphone Recording</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        duration = st.slider(
            "⏱️ Recommended Recording Duration (seconds)",
            min_value=3,
            max_value=10,
            value=5
        )

    with col2:
        st.markdown("""
        <div class="feature-info">
        🎤 <b>Recording Instructions:</b><br>
        1. Click "Start Recording"<br>
        2. Say "Aah" in a steady, continuous voice<br>
        3. Click "Stop Recording"<br>
        4. Analysis starts automatically
        </div>
        """, unsafe_allow_html=True)

    audio = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        use_container_width=True
    )

    if audio:

    audio_bytes = audio["bytes"]

    audio_segment = AudioSegment.from_file(
        io.BytesIO(audio_bytes),
        format="webm"
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
        audio_segment.export(tmp_path, format="wav")

    st.audio(audio_bytes, format="audio/webm")

    status_text = st.empty()
    status_text.markdown("✅ **Recording done!** Analyzing...")

    with st.spinner("📈 Processing waveform..."):
        fig = plot_waveform(tmp_path)
        st.pyplot(fig)
        plt.close()

    with st.spinner("⏳ Extracting voice features..."):
        features, error = extract_features_from_audio(tmp_path)

    if error:
        st.error(f"❌ {error}")

    else:
        with st.spinner("🤖 Running prediction models..."):
            results = predict(features)

        show_results(results, features, "Live Microphone Recording")

    os.unlink(tmp_path)


# ══════════════════════════════════════════════════
#  MODE: MANUAL INPUT
# ══════════════════════════════════════════════════
elif mode == "✍️ Manual Input":
    st.markdown('<div class="section-title">✍️ Enter Feature Values Manually</div>', unsafe_allow_html=True)

    defaults = {
        'MDVP:Fo(Hz)': 154.23, 'MDVP:Fhi(Hz)': 197.10, 'MDVP:Flo(Hz)': 116.32,
        'MDVP:Jitter(%)': 0.00622, 'MDVP:Jitter(Abs)': 0.000044, 'MDVP:RAP': 0.00307,
        'MDVP:PPQ': 0.00345, 'Jitter:DDP': 0.00920, 'MDVP:Shimmer': 0.02971,
        'MDVP:Shimmer(dB)': 0.282, 'Shimmer:APQ3': 0.01491, 'Shimmer:APQ5': 0.01941,
        'MDVP:APQ': 0.02462, 'Shimmer:DDA': 0.04473, 'NHR': 0.01160,
        'HNR': 21.886, 'RPDE': 0.498, 'DFA': 0.718, 'spread1': -5.684,
        'spread2': 0.227, 'D2': 2.382, 'PPE': 0.207
    }
    st.info("💡 Enter patient feature values, or use the default values for testing")

    input_values = {}
    groups = [
        ("🎵 Voice Frequencies",            ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)']),
        ("📳 Jitter (Frequency Variation)",  ['MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP']),
        ("📉 Shimmer (Amplitude Variation)", ['MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA']),
        ("🔊 Noise Ratios",                  ['NHR', 'HNR']),
        ("🔬 Nonlinear Measures",            ['RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'])
    ]

    for group_title, feat_list in groups:
        st.markdown(f"**{group_title}**")
        cols = st.columns(len(feat_list) if len(feat_list) <= 3 else 3)
        for i, feat in enumerate(feat_list):
            col_idx = i % 3 if len(feat_list) > 3 else i
            with cols[col_idx]:
                input_values[feat] = st.number_input(
                    feat, value=float(defaults[feat]), format="%.6f",
                    help=FEATURE_DESCRIPTIONS.get(feat, ""), key=f"input_{feat}")
        st.markdown("---")

    if st.button("🔍 Analyze & Predict", type="primary"):
        features = np.array([input_values[f] for f in FEATURE_NAMES])
        results  = predict(features)
        show_results(results, features, "Manual Input")


# ══════════════════════════════════════════════════
#  MODE: ABOUT
# ══════════════════════════════════════════════════
elif mode == "ℹ️ About":
    st.markdown('<div class="section-title">ℹ️ About the Project</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Project Goal
        Develop an AI system capable of **early detection** of Parkinson's disease
        through voice characteristic analysis using two ensemble algorithms:
        - **Extra Trees** (Best Model — AUC 99.14%)
        - **Random Forest** (AUC 96.55%)

        ### 📊 Dataset Used
        - **UCI Parkinson's Dataset**
        - 195 voice recordings from 31 people
        - 23 voice variables (22 features + status)
        - 147 Parkinson's cases, 48 healthy cases

        ### ⚖️ Class Imbalance Handling
        - **SMOTE** (Synthetic Minority Over-sampling)
        applied during training to balance classes

        ### 🔬 Extracted Features
        - **Voice frequencies**: Fo, Fhi, Flo
        - **Frequency variation**: Jitter (5 measures)
        - **Amplitude variation**: Shimmer (6 measures)
        - **Noise ratios**: NHR, HNR
        - **Nonlinear measures**: RPDE, DFA, D2, PPE
        """)

    with col2:
        st.markdown("""
        ### 🤖 Models Used

        #### Extra Trees Classifier ⭐ Best
        An ensemble of extremely randomized decision trees.
        Introduces more randomness than Random Forest by
        also randomizing split thresholds — leading to
        lower variance and often better generalization.

        #### Random Forest
        An ensemble of decision trees trained on random
        subsets of data and features (bagging). Robust,
        interpretable, and excellent for clinical data.

        #### Training Pipeline
        `StandardScaler → SMOTE → Model`
        - 10-Fold Stratified Cross-Validation
        - GridSearchCV for hyperparameter tuning
        - Evaluation: Accuracy, F1, AUC-ROC

        ### ⚠️ Disclaimer
        This system is for **research and academic purposes** only.
        Results are **NOT a medical diagnosis**.
        Always consult a specialized physician.
        """)

    st.divider()
    st.markdown("### 📈 Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="metric-card"><div class="metric-value">99.14%</div><div class="metric-label">Extra Trees AUC</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><div class="metric-value">96.55%</div><div class="metric-label">Random Forest AUC</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><div class="metric-value">10-Fold</div><div class="metric-label">Cross Validation</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card"><div class="metric-value">195</div><div class="metric-label">Dataset Size</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; color:#555; font-size:0.8rem; margin-top:50px; padding:20px;
     border-top:1px solid #2a3f5f;'>
    🧠 Parkinson's Disease Detection System | Graduation Project 2026<br>
    ⚠️ For research and academic purposes only — Not a medical diagnosis
</div>
""", unsafe_allow_html=True)
