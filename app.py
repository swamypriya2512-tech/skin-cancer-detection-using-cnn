"""
Skin Cancer Identification — Premium Streamlit Application
XceptionNet + EfficientNetB0 + PSO + SVM/RF/KNN/NB + Ensemble + GradCAM + LIME
"""
import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (
    MODEL_DIR, RESULT_DIR, METRICS_PATH, CV_RESULTS_PATH,
    ENCODER_PATH, SCALER_PATH, PSO_MASK_PATH,
    SVM_PATH, RF_PATH, KNN_PATH, NB_PATH, ENSEMBLE_PATH,
    IMG_SIZE, CLASS_NAMES
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermAI — Skin Cancer Identification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  — Premium Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #0a0e1a; }
  .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
  }
  [data-testid="stSidebar"] .stMarkdown { color: #c9d6e3; }

  /* ── Hero Banner ── */
  .hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 40%, #0d2137 100%);
    border: 1px solid #1e4d7b;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,100,255,0.12);
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #e879f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
  }
  .hero-sub {
    color: #7fa8c9; font-size: 0.95rem; margin: 0;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8; font-size: 0.75rem;
    padding: 3px 10px; border-radius: 20px; margin: 0.6rem 4px 0 0;
  }

  /* ── Metric Cards ── */
  .metric-card {
    background: linear-gradient(135deg, #111827, #1a2744);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(56,189,248,0.15);
  }
  .metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .metric-label { color: #7fa8c9; font-size: 0.82rem; margin-top: 0.2rem; }

  /* ── Section headers ── */
  .section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem; font-weight: 600;
    color: #e2e8f0;
    padding: 0.5rem 0 0.8rem 0;
    border-bottom: 2px solid #1e3a5f;
    margin-bottom: 1rem;
  }

  /* ── Step pills ── */
  .step-pill {
    display: inline-block;
    background: linear-gradient(135deg, #1e4d7b, #1a2744);
    border: 1px solid #2563eb;
    color: #93c5fd; font-size: 0.78rem; font-weight: 600;
    padding: 4px 14px; border-radius: 20px; margin-bottom: 1rem;
  }

  /* ── Prediction result card ── */
  .pred-card {
    background: linear-gradient(135deg, #0f1b35 0%, #1a2744 100%);
    border: 1px solid #1e4d7b;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
  }
  .pred-class {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem; font-weight: 700; color: #38bdf8;
  }
  .pred-conf { color: #94a3b8; font-size: 0.9rem; }

  /* ── Warning cards ── */
  .warning-malignant {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 10px; padding: 0.8rem 1rem;
    color: #fca5a5; font-size: 0.88rem;
  }
  .warning-benign {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 10px; padding: 0.8rem 1rem;
    color: #86efac; font-size: 0.88rem;
  }

  /* ── Progress bar colour ── */
  .stProgress > div > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
  }

  /* ── Tab style ── */
  .stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500; font-size: 0.9rem;
    color: #7fa8c9;
  }
  .stTabs [aria-selected="true"] { color: #38bdf8 !important; }

  /* ── Generic text ── */
  h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif; }
  p, li { color: #c9d6e3; }
  label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Disease Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "akiec": {
        "name": "Actinic Keratosis (Pre-cancerous)",
        "description": "Actinic keratoses (Solar keratoses) and intraepithelial carcinoma (Bowen's disease) are common non-invasive variants of squamous cell carcinoma. They appear as scaly, crusty lesions on sun-damaged skin.",
        "risk": "High Risk — Pre-cancerous",
        "action": "Needs clinical monitoring and likely treatment (cryotherapy, topical creams, or minor surgery) to prevent progression to invasive carcinoma."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma (Malignant)",
        "description": "Basal Cell Carcinoma (BCC) is the most common form of skin cancer. It rarely metastasizes but grows destructively if untreated. It often appears as a transparent bump, a pearly nodule, or a sore that doesn't heal.",
        "risk": "High Risk — Malignant",
        "action": "Immediate dermatological intervention is required. Fully curable if surgically removed early."
    },
    "bkl": {
        "name": "Benign Keratosis",
        "description": "Benign keratosis is a generic class that includes seborrheic keratoses (wart-like growths), solar lentigines (liver spots or age spots), and lichen-planus like keratoses. These lesions are perfectly harmless.",
        "risk": "Low Risk — Benign",
        "action": "No treatment necessary unless it causes discomfort or cosmetic concerns."
    },
    "df": {
        "name": "Dermatofibroma",
        "description": "Dermatofibroma is a benign skin lesion that often appears after a minor injury, like a bug bite or a splinter. They are firm bumps, often feeling like a small hard stone under the skin.",
        "risk": "Low Risk — Benign",
        "action": "Harmless; treatment is usually not required unless symptoms arise."
    },
    "mel": {
        "name": "Melanoma (Malignant)",
        "description": "Melanoma is the most dangerous form of skin cancer, originating from pigment-producing cells (melanocytes). It can grow aggressively and spread to other organs if not caught early. Look out for the 'ABCDE' warning signs (Asymmetry, Border, Color, Diameter, Evolving).",
        "risk": "Critical Risk — Malignant",
        "action": "URGENT dermatological evaluation and surgical excision. Early detection drastically improves survival rates."
    },
    "nv": {
        "name": "Melanocytic Nevi (Moles)",
        "description": "Melanocytic nevi are common, benign moles. They are usually uniformly colored (brown, tan, or black) and have regular borders. Almost everyone has them.",
        "risk": "Low Risk — Benign",
        "action": "Generally harmless. Routine monitoring is recommended; seek advice if it changes size, shape, or bleeds."
    },
    "vasc": {
        "name": "Vascular Lesions",
        "description": "Vascular lesions involve blood vessels and include cherry angiomas, hemangiomas, and angiokeratomas. They typically appear as bright red or purple papules.",
        "risk": "Low Risk — Benign",
        "action": "Usually completely harmless. Laser therapy can be considered for cosmetic removal if desired."
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}

def full_class_name(abbr: str) -> str:
    return CLASS_NAMES.get(abbr, abbr)

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all saved model artifacts."""
    artifacts = {}
    try:
        artifacts["encoder"]  = joblib.load(ENCODER_PATH)
        artifacts["scaler"]   = joblib.load(SCALER_PATH)
        artifacts["pso_mask"] = np.load(PSO_MASK_PATH)
        for name, path in [("SVM", SVM_PATH), ("Random Forest", RF_PATH),
                            ("KNN", KNN_PATH)]:
            if os.path.exists(path):
                artifacts[name] = joblib.load(path)
        if os.path.exists(ENSEMBLE_PATH):
            artifacts["Ensemble"] = joblib.load(ENSEMBLE_PATH)
    except Exception as e:
        st.warning(f"Some models could not be loaded: {e}")
    return artifacts


@st.cache_resource(show_spinner=False)
def load_cnn_models():
    """Load Xception + EfficientNet feature extractors."""
    import tensorflow as tf
    from tensorflow.keras.applications import Xception, EfficientNetB0
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    base_xc = Xception(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_xc.trainable = False
    xc_model = Model(base_xc.input, GlobalAveragePooling2D()(base_xc.output))

    base_en = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_en.trainable = False
    en_model = Model(base_en.input, GlobalAveragePooling2D()(base_en.output))

    return xc_model, en_model


def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL → (1, 224, 224, 3) float32 array."""
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def extract_image_features(img_array, xc_model, en_model, scaler, pso_mask):
    """Full inference-time feature extraction pipeline."""
    xc_f = xc_model.predict(img_array, verbose=0)
    en_f = en_model.predict(img_array, verbose=0)
    feat = np.concatenate([xc_f, en_f], axis=1)
    feat = scaler.transform(feat)
    feat = feat[:, pso_mask]
    return feat


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def load_cv_results():
    if os.path.exists(CV_RESULTS_PATH):
        with open(CV_RESULTS_PATH) as f:
            return json.load(f)
    return {}


def models_trained() -> bool:
    required = [ENCODER_PATH, SCALER_PATH, PSO_MASK_PATH, RF_PATH]
    return all(os.path.exists(p) for p in required)


def confidence_bar(label, prob, color="#38bdf8"):
    pct = prob * 100
    filled = int(pct / 2)
    empty  = 50 - filled
    bar = "█" * filled + "░" * empty
    return f"**{label}** `{pct:.1f}%`\n`{bar}`"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
      <div style='font-size:2.5rem'>🔬</div>
      <div style='font-family:Space Grotesk;font-size:1.1rem;font-weight:700;
                  background:linear-gradient(135deg,#38bdf8,#818cf8);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        DermAI
      </div>
      <div style='color:#7fa8c9;font-size:0.75rem'>Skin Cancer Identification System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Select Module",
        ["🏠 Dashboard", "⚙️ Train Models", "🔍 Predict & Analyse",
         "📊 Classifier Comparison", "🧬 Cross-Validation", "📖 About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🔧 PSO Settings")
    pso_particles = st.slider("Particles", 5, 30, 20)
    pso_iters     = st.slider("Iterations", 5, 50, 30)
    pso_k         = st.slider("Features to select (k)", 50, 500, 300)

    st.markdown("---")
    status_color = "#22c55e" if models_trained() else "#f59e0b"
    status_text  = "Models Ready ✓" if models_trained() else "Models Not Trained"
    st.markdown(f"""
    <div style='background:rgba(0,0,0,0.3);border:1px solid {status_color}33;
                border-radius:8px;padding:0.7rem;text-align:center;'>
      <span style='color:{status_color};font-size:0.85rem;font-weight:600;'>{status_text}</span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">🔬 DermAI — Skin Cancer Identification</p>
  <p class="hero-sub">CNN + PSO Feature Selection + Ensemble ML + Grad-CAM + LIME Explainability</p>
  <span class="hero-badge">XceptionNet</span>
  <span class="hero-badge">EfficientNetB0</span>
  <span class="hero-badge">PSO</span>
  <span class="hero-badge">SVM</span>
  <span class="hero-badge">Random Forest</span>
  <span class="hero-badge">KNN</span>
  <span class="hero-badge">Ensemble</span>
  <span class="hero-badge">Grad-CAM</span>
  <span class="hero-badge">LIME</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    metrics = load_metrics()
    cv_data = load_cv_results()

    if not metrics:
        st.info("⚠️ No trained models found. Go to **⚙️ Train Models** to start training.")
    else:
        st.markdown('<div class="section-header">📈 Model Performance Overview</div>', unsafe_allow_html=True)

        # Top metric cards
        clf_names  = [k for k in metrics if k != "Ensemble"]
        best_name  = max(clf_names, key=lambda k: metrics[k].get("accuracy", 0))
        best_acc   = metrics[best_name]["accuracy"]
        ens_acc    = metrics.get("Ensemble", {}).get("accuracy", best_acc)
        best_f1    = metrics[best_name]["f1_score"]
        n_classes  = 7 if not os.path.exists(ENCODER_PATH) else len(joblib.load(ENCODER_PATH).classes_)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, f"{best_acc*100:.1f}%",  "Best Classifier Accuracy"),
            (c2, f"{ens_acc*100:.1f}%",   "Ensemble Accuracy"),
            (c3, f"{best_f1*100:.1f}%",   "Best F1-Score"),
            (c4, f"{n_classes}",           "Disease Classes"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Accuracy bar chart
        st.markdown('<div class="section-header">🏆 Classifier Accuracy Comparison</div>', unsafe_allow_html=True)
        names   = list(metrics.keys())
        accs    = [metrics[n].get("accuracy", 0) * 100 for n in names]
        colors  = ["#38bdf8" if n == "Ensemble" else "#818cf8" for n in names]

        fig = go.Figure(go.Bar(
            x=names, y=accs,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{a:.1f}%" for a in accs],
            textposition="outside",
            textfont=dict(color="white", size=13),
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#111827",
            font=dict(color="#c9d6e3", family="Inter"),
            yaxis=dict(range=[0, 105], showgrid=True, gridcolor="#1e3a5f"),
            xaxis=dict(showgrid=False),
            margin=dict(t=30, b=20),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-classifier detail
        st.markdown('<div class="section-header">📋 Detailed Metrics</div>', unsafe_allow_html=True)
        rows = []
        for n, m in metrics.items():
            rows.append({
                "Classifier":  n,
                "Accuracy":    f"{m.get('accuracy',0)*100:.2f}%",
                "Precision":   f"{m.get('precision',0)*100:.2f}%",
                "Recall":      f"{m.get('recall',0)*100:.2f}%",
                "F1-Score":    f"{m.get('f1_score',0)*100:.2f}%",
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        # Confusion matrix for best classifier
        if "confusion_matrix" in metrics.get(best_name, {}):
            st.markdown(f'<div class="section-header">🧩 Confusion Matrix — {best_name}</div>',
                        unsafe_allow_html=True)
            cm       = np.array(metrics[best_name]["confusion_matrix"])
            encoder  = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None
            labels   = list(encoder.classes_) if encoder else [str(i) for i in range(cm.shape[0])]

            fig2 = px.imshow(
                cm, text_auto=True,
                x=labels, y=labels,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual"),
            )
            fig2.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#c9d6e3"),
                height=420,
                margin=dict(t=30, b=20),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── HAM10000 Class info ──────────────────────────────────────────
    st.markdown('<div class="section-header">🏥 HAM10000 Disease Classes</div>', unsafe_allow_html=True)
    class_data = [
        ("akiec", "Actinic Keratosis",   "Pre-cancerous",  "⚠️", "#f59e0b"),
        ("bcc",   "Basal Cell Carcinoma","Malignant",       "🔴", "#ef4444"),
        ("bkl",   "Benign Keratosis",    "Benign",          "🟢", "#22c55e"),
        ("df",    "Dermatofibroma",      "Benign",          "🟢", "#22c55e"),
        ("mel",   "Melanoma",            "Malignant",       "🔴", "#ef4444"),
        ("nv",    "Melanocytic Nevi",    "Benign",          "🟢", "#22c55e"),
        ("vasc",  "Vascular Lesions",    "Rare",            "🟡", "#a855f7"),
    ]
    cols = st.columns(4)
    for i, (code, name, cat, ico, clr) in enumerate(class_data):
        cols[i % 4].markdown(f"""
        <div style='background:#111827;border:1px solid {clr}33;border-left:3px solid {clr};
                    border-radius:8px;padding:0.7rem 0.9rem;margin-bottom:0.5rem;'>
          <div style='font-size:0.7rem;color:#7fa8c9;font-weight:600;text-transform:uppercase'>
            {code.upper()}
          </div>
          <div style='color:#e2e8f0;font-weight:600;font-size:0.88rem'>{name}</div>
          <div style='color:{clr};font-size:0.75rem'>{ico} {cat}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Train Models
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚙️ Train Models":
    st.markdown('<div class="section-header">⚙️ Train All Models</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:1rem 1.5rem;margin-bottom:1rem;'>
      <b style='color:#38bdf8'>Training Pipeline:</b>
      <ol style='color:#c9d6e3;margin:0.5rem 0 0 0;'>
        <li>Load HAM10000 images + labels from metadata XLSX</li>
        <li>Extract features with <b>XceptionNet</b> + <b>EfficientNetB0</b> (ImageNet pretrained)</li>
        <li>Scale features with MinMax Scaler</li>
        <li>Run <b>PSO</b> (Particle Swarm Optimisation) to select best features</li>
        <li>Train <b>SVM</b>, <b>Random Forest</b>, and <b>KNN</b></li>
        <li>Build <b>Soft-Voting Ensemble</b></li>
        <li>Evaluate + run <b>K-Fold</b> and <b>Cross-Dataset</b> validation</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        force_retrain = st.checkbox("Force re-extract features (slow)", value=False)
        skip_pso      = st.checkbox("Skip PSO (use SelectKBest fallback — faster)", value=False)
    with col_b:
        train_btn = st.button("🚀 Start Training", type="primary", use_container_width=True)

    if train_btn:
        # ── override config from sidebar sliders ──
        import src.config as cfg
        cfg.PSO_N_PARTICLES = pso_particles
        cfg.PSO_ITERS       = pso_iters
        cfg.PSO_K_FEATURES  = pso_k

        if force_retrain:
            for p in [cfg.FEAT_TRAIN_PATH, cfg.FEAT_TEST_PATH,
                      cfg.LABEL_TRAIN_PATH, cfg.LABEL_TEST_PATH, cfg.PSO_MASK_PATH]:
                if os.path.exists(p): os.remove(p)

        if skip_pso:
            import src.pso_selector as pso_mod
            pso_mod.PSO_AVAILABLE = False

        status_box = st.empty()
        prog_bar   = st.progress(0)
        log_box    = st.empty()
        log_lines  = []

        def progress_callback(msg, pct=None):
            status_box.markdown(f"""
            <div style='background:#111827;border:1px solid #1e4d7b;border-radius:8px;
                        padding:0.6rem 1rem;color:#38bdf8;font-size:0.88rem;'>
              ⏳ {msg}
            </div>""", unsafe_allow_html=True)
            if pct is not None:
                prog_bar.progress(min(float(pct), 1.0))
            log_lines.append(f"• {msg}")
            log_box.markdown(
                "<div style='background:#0a0e1a;border:1px solid #1e3a5f;"
                "border-radius:8px;padding:0.8rem;max-height:200px;overflow-y:auto;"
                "font-size:0.78rem;color:#7fa8c9;'>" +
                "<br>".join(log_lines[-12:]) + "</div>",
                unsafe_allow_html=True
            )

        try:
            from src.train import run_training
            t0 = time.time()
            progress_callback("Starting training pipeline …", 0.02)
            metrics, cv_results, class_names, pso_mask = run_training(
                progress_callback=progress_callback
            )
            elapsed = time.time() - t0
            prog_bar.progress(1.0)

            st.success(f"✅ Training complete in {elapsed/60:.1f} minutes!")

            # Show results
            st.markdown('<div class="section-header">📊 Training Results</div>',
                        unsafe_allow_html=True)
            rows = []
            for name, m in metrics.items():
                rows.append({
                    "Classifier": name,
                    "Accuracy":   f"{m.get('accuracy',0)*100:.2f}%",
                    "F1-Score":   f"{m.get('f1_score',0)*100:.2f}%",
                    "Precision":  f"{m.get('precision',0)*100:.2f}%",
                    "Recall":     f"{m.get('recall',0)*100:.2f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Radar chart
            clf_metrics = {k: v for k, v in metrics.items() if k != "Ensemble"}
            categories  = ["Accuracy", "Precision", "Recall", "F1-Score"]
            fig = go.Figure()
            colors_radar = [
                "rgba(56, 189, 248, 0.2)",  # #38bdf8
                "rgba(129, 140, 248, 0.2)", # #818cf8
                "rgba(232, 121, 249, 0.2)", # #e879f9
                "rgba(52, 211, 153, 0.2)",  # #34d399
                "rgba(245, 158, 11, 0.2)",  # #f59e0b
            ]
            colors_line = ["#38bdf8", "#818cf8", "#e879f9", "#34d399", "#f59e0b"]
            
            for i, (name, m) in enumerate(clf_metrics.items()):
                vals = [
                    m.get("accuracy", 0) * 100,
                    m.get("precision", 0) * 100,
                    m.get("recall", 0) * 100,
                    m.get("f1_score", 0) * 100,
                ]
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=categories + [categories[0]],
                    fill="toself", name=name,
                    line=dict(color=colors_line[i % len(colors_line)], width=2),
                    fillcolor=colors_radar[i % len(colors_radar)],
                    opacity=0.85,
                ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color="#7fa8c9"),
                    angularaxis=dict(color="#7fa8c9"),
                    bgcolor="#111827",
                ),
                paper_bgcolor="#0d1117",
                font=dict(color="#c9d6e3"),
                showlegend=True,
                height=400,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Predict & Analyse
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Predict & Analyse":
    st.markdown('<div class="section-header">🔍 Upload & Analyse Skin Lesion</div>', unsafe_allow_html=True)

    if not models_trained():
        st.error("⚠️ No trained models found. Please train models first.")
        st.stop()

    artifacts  = load_models()
    encoder    = artifacts.get("encoder")
    scaler     = artifacts.get("scaler")
    pso_mask   = artifacts.get("pso_mask")
    class_list = list(encoder.classes_) if encoder else []

    uploaded = st.file_uploader(
        "Upload a dermoscopy image (JPG/PNG)", type=["jpg", "jpeg", "png"],
        help="Upload a skin lesion dermoscopy image for analysis"
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        img_arr = preprocess_pil(pil_img)

        col_img, col_res = st.columns([1, 1.6])

        with col_img:
            st.markdown('<div class="section-header">🖼️ Input Image</div>', unsafe_allow_html=True)
            st.image(pil_img, use_column_width=True, caption="Uploaded lesion image")

        with col_res:
            st.markdown('<div class="section-header">⚡ Running Analysis …</div>', unsafe_allow_html=True)
            with st.spinner("Extracting CNN features …"):
                xc_model, en_model = load_cnn_models()
                feat = extract_image_features(img_arr, xc_model, en_model, scaler, pso_mask)

            # Predict with all classifiers
            predictions = {}
            for name in ["SVM", "Random Forest", "KNN"]:
                if name in artifacts:
                    clf = artifacts[name]
                    pred_idx  = clf.predict(feat)[0]
                    pred_prob = clf.predict_proba(feat)[0] if hasattr(clf, "predict_proba") else None
                    pred_label = encoder.inverse_transform([pred_idx])[0]
                    top_conf   = float(pred_prob[pred_idx]) if pred_prob is not None else 0.5
                    predictions[name] = {
                        "label": pred_label,
                        "conf":  top_conf,
                        "probs": pred_prob,
                    }

            # Ensemble prediction (average proba)
            all_probas = [p["probs"] for p in predictions.values() if p["probs"] is not None]
            if all_probas:
                avg_proba   = np.mean(all_probas, axis=0)
                ens_idx     = np.argmax(avg_proba)
                ens_label   = encoder.inverse_transform([ens_idx])[0]
                ens_conf    = float(avg_proba[ens_idx])
            else:
                ens_label = "Unknown"
                ens_conf  = 0.0
                avg_proba = np.zeros(len(class_list))

            is_malignant = ens_label in MALIGNANT_CLASSES
            warning_class = "warning-malignant" if is_malignant else "warning-benign"
            warning_msg   = (
                "🔴 High Risk: This lesion shows characteristics consistent with a <b>malignant</b> condition. "
                "Please consult a dermatologist immediately."
                if is_malignant else
                "🟢 Low Risk: This lesion appears <b>benign</b>. Regular monitoring is still recommended."
            )

            st.markdown(f"""
            <div class="pred-card">
              <div style='color:#7fa8c9;font-size:0.8rem;text-transform:uppercase;letter-spacing:2px;'>
                Ensemble Prediction
              </div>
              <div class="pred-class">{full_class_name(ens_label)}</div>
              <div style='color:#38bdf8;font-size:1.1rem;font-weight:600;'>
                ({ens_label.upper()}) — {ens_conf*100:.1f}% confidence
              </div>
            </div>
            <br>
            <div class="{warning_class}">{warning_msg}</div>
            """, unsafe_allow_html=True)
            
            # --- Disease Information Box ---
            info = DISEASE_INFO.get(ens_label, {})
            if info:
                st.markdown(f"""
                <div style='background:#111827;border-left:4px solid {"#ef4444" if is_malignant else "#22c55e"};
                            border-radius:4px;padding:1rem;margin-top:1rem;'>
                    <h4 style='color:#e2e8f0;margin-top:0;'>🩺 What is {info['name']}?</h4>
                    <p style='color:#94a3b8;font-size:0.9rem;'>{info['description']}</p>
                    <p style='color:#c9d6e3;font-size:0.9rem;'><b>Risk Level:</b> {info['risk']}</p>
                    <p style='color:#c9d6e3;font-size:0.9rem;'><b>Recommended Action:</b> {info['action']}</p>
                </div>
                """, unsafe_allow_html=True)

        # ── Per-class probability chart ──────────────────────────────
        st.markdown('<div class="section-header">📊 Class Probability Distribution</div>',
                    unsafe_allow_html=True)
        sorted_idx  = np.argsort(avg_proba)[::-1]
        sorted_lbls = [full_class_name(class_list[i]) for i in sorted_idx]
        sorted_prob = [avg_proba[i] * 100 for i in sorted_idx]
        bar_colors  = ["#ef4444" if class_list[i] in MALIGNANT_CLASSES else "#38bdf8"
                       for i in sorted_idx]

        fig = go.Figure(go.Bar(
            y=sorted_lbls, x=sorted_prob, orientation="h",
            marker=dict(color=bar_colors),
            text=[f"{p:.1f}%" for p in sorted_prob],
            textposition="outside",
            textfont=dict(color="white"),
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#111827",
            font=dict(color="#c9d6e3"),
            xaxis=dict(range=[0, 110], showgrid=True, gridcolor="#1e3a5f", title="Probability (%)"),
            yaxis=dict(showgrid=False, autorange="reversed"),
            height=350, margin=dict(t=20, b=20, l=20, r=60),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Classifier agreement table ───────────────────────────────
        st.markdown('<div class="section-header">🤝 Classifier Predictions</div>',
                    unsafe_allow_html=True)
        agree_rows = []
        for n, p in predictions.items():
            agree_rows.append({
                "Classifier":   n,
                "Prediction":   full_class_name(p["label"]),
                "Code":         p["label"].upper(),
                "Confidence":   f"{p['conf']*100:.1f}%",
                "Risk":         "🔴 Malignant" if p["label"] in MALIGNANT_CLASSES else "🟢 Benign/Low",
            })
        st.dataframe(pd.DataFrame(agree_rows), use_container_width=True, hide_index=True)

        # ── Grad-CAM ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">🌡️ Grad-CAM Heatmap (Xception)</div>',
                    unsafe_allow_html=True)
        with st.spinner("Generating Grad-CAM …"):
            try:
                from src.gradcam import compute_gradcam
                gc_img, heatmap = compute_gradcam(img_arr)
                col_gc1, col_gc2 = st.columns(2)
                col_gc1.image(np.uint8(img_arr[0] * 255), caption="Original", use_column_width=True)
                col_gc2.image(gc_img, caption="Grad-CAM Overlay", use_column_width=True)

                # Heatmap intensity bar
                fig_hm = go.Figure(go.Heatmap(
                    z=cv2.resize(heatmap, (64, 64)),
                    colorscale="Jet", showscale=True,
                ))
                fig_hm.update_layout(
                    title="Attention Heatmap", paper_bgcolor="#0d1117",
                    plot_bgcolor="#111827", font=dict(color="#c9d6e3"),
                    height=280, margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig_hm, use_container_width=True)
                st.info("🔴 **Red/Yellow regions** = areas the model focused on most. "
                        "These highlight the most diagnostically relevant features.")
            except Exception as e:
                st.warning(f"Grad-CAM could not be generated: {e}")

        # ── LIME ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🧩 LIME Explainability</div>',
                    unsafe_allow_html=True)
        run_lime = st.button("Generate LIME Explanation (takes ~30s)", key="lime_btn")
        if run_lime:
            with st.spinner("Running LIME superpixel analysis …"):
                try:
                    from src.xai import explain_with_lime
                    explanation, img_boundary = explain_with_lime(
                        img_arr[0], xc_model, en_model, scaler, pso_mask,
                        artifacts.get("Random Forest") or artifacts.get("SVM"),
                        num_samples=200,
                    )
                    if img_boundary is not None:
                        cl1, cl2 = st.columns(2)
                        cl1.image(np.uint8(img_arr[0] * 255), caption="Original", use_column_width=True)
                        cl2.image(img_boundary, caption="LIME: Positive Superpixels", use_column_width=True)
                        st.success("✅ Green regions = superpixels most responsible for the prediction.")
                    else:
                        st.warning("LIME explanation unavailable.")
                except Exception as e:
                    st.error(f"LIME error: {e}")

        # ── RF Feature Importance ────────────────────────────────────
        if "Random Forest" in artifacts:
            st.markdown('<div class="section-header">🧠 Top Feature Importances (Random Forest)</div>',
                        unsafe_allow_html=True)
            rf_model = artifacts["Random Forest"]
            from src.xai import get_feature_importance_from_rf
            importances = get_feature_importance_from_rf(rf_model, top_n=20)
            fi_names = [f"Feature {i+1}" for i, _ in enumerate(importances)]
            fi_vals  = [v * 100 for _, v in importances]

            fig_fi = go.Figure(go.Bar(
                x=fi_names, y=fi_vals,
                marker=dict(
                    color=fi_vals,
                    colorscale="Plasma",
                    showscale=False,
                ),
                text=[f"{v:.2f}%" for v in fi_vals],
                textposition="outside",
            ))
            fig_fi.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#111827",
                font=dict(color="#c9d6e3"),
                yaxis=dict(title="Importance (%)", showgrid=True, gridcolor="#1e3a5f"),
                xaxis=dict(showgrid=False, tickangle=-45),
                height=340, margin=dict(t=20, b=60),
            )
            st.plotly_chart(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Classifier Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Classifier Comparison":
    st.markdown('<div class="section-header">📊 Classifier Performance Comparison</div>',
                unsafe_allow_html=True)

    metrics = load_metrics()
    if not metrics:
        st.info("⚠️ No trained models found.")
        st.stop()

    clf_names = list(metrics.keys())
    accs  = [metrics[n].get("accuracy", 0) * 100  for n in clf_names]
    precs = [metrics[n].get("precision", 0) * 100 for n in clf_names]
    recs  = [metrics[n].get("recall", 0) * 100    for n in clf_names]
    f1s   = [metrics[n].get("f1_score", 0) * 100  for n in clf_names]

    # Grouped bar
    fig = go.Figure()
    for vals, label, color in [
        (accs,  "Accuracy",  "#38bdf8"),
        (precs, "Precision", "#818cf8"),
        (recs,  "Recall",    "#e879f9"),
        (f1s,   "F1-Score",  "#34d399"),
    ]:
        fig.add_trace(go.Bar(name=label, x=clf_names, y=vals, marker_color=color,
                             text=[f"{v:.1f}" for v in vals], textposition="outside",
                             textfont=dict(color="white", size=11)))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="#0d1117", plot_bgcolor="#111827",
        font=dict(color="#c9d6e3", family="Inter"),
        yaxis=dict(range=[0, 110], showgrid=True, gridcolor="#1e3a5f", title="Score (%)"),
        xaxis=dict(showgrid=False),
        legend=dict(bgcolor="#111827", bordercolor="#1e3a5f"),
        height=420, margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: Accuracy vs F1
    st.markdown('<div class="section-header">⚖️ Accuracy vs F1-Score</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    colors_s = ["#38bdf8", "#818cf8", "#e879f9", "#34d399", "#f59e0b"]
    for i, n in enumerate(clf_names):
        m = metrics[n]
        fig2.add_trace(go.Scatter(
            x=[m.get("accuracy", 0) * 100],
            y=[m.get("f1_score", 0) * 100],
            mode="markers+text",
            name=n,
            text=[n],
            textposition="top center",
            marker=dict(size=18, color=colors_s[i % len(colors_s)],
                        line=dict(width=2, color="white")),
        ))
    fig2.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#111827",
        font=dict(color="#c9d6e3"),
        xaxis=dict(title="Accuracy (%)", showgrid=True, gridcolor="#1e3a5f"),
        yaxis=dict(title="F1-Score (%)", showgrid=True, gridcolor="#1e3a5f"),
        height=380, margin=dict(t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Per-class F1 heatmap
    if encoder := (joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None):
        class_labels = list(encoder.classes_)
        st.markdown('<div class="section-header">📐 Per-Class F1 Heatmap</div>',
                    unsafe_allow_html=True)
        heat_data = []
        for n, m in metrics.items():
            cr = m.get("classification_report", {})
            row = [cr.get(lbl, {}).get("f1-score", 0) * 100 for lbl in class_labels]
            heat_data.append(row)

        fig3 = px.imshow(
            heat_data, text_auto=".1f",
            x=class_labels, y=list(metrics.keys()),
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=100,
            labels=dict(color="F1 (%)"),
        )
        fig3.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#c9d6e3"),
            height=300, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧬 Cross-Validation":
    st.markdown('<div class="section-header">🧬 Cross-Validation Results</div>',
                unsafe_allow_html=True)

    cv_data = load_cv_results()
    if not cv_data:
        st.info("⚠️ No cross-validation data found. Train models first.")
        st.stop()

    # ── K-Fold results ───────────────────────────────────────────────
    st.markdown("### 🔁 Stratified K-Fold Cross-Validation")
    kfold = cv_data.get("kfold", {})
    if kfold:
        cv_rows = []
        for clf_n, d in kfold.items():
            cv_rows.append({
                "Classifier":    clf_n,
                "Mean Accuracy": f"{d['mean_accuracy']*100:.2f}%",
                "Std Dev":       f"± {d['std_accuracy']*100:.2f}%",
                "Mean F1":       f"{d['mean_f1']*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

        # Per-fold accuracy line chart
        fig = go.Figure()
        colors_cv = ["#38bdf8", "#818cf8"]
        for i, (clf_n, d) in enumerate(kfold.items()):
            folds = list(range(1, len(d["fold_accuracies"]) + 1))
            accs  = [a * 100 for a in d["fold_accuracies"]]
            fig.add_trace(go.Scatter(
                x=folds, y=accs, name=clf_n, mode="lines+markers",
                line=dict(color=colors_cv[i], width=2.5),
                marker=dict(size=8),
            ))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#111827",
            font=dict(color="#c9d6e3"),
            xaxis=dict(title="Fold", showgrid=True, gridcolor="#1e3a5f",
                       tickvals=list(range(1, 6))),
            yaxis=dict(title="Accuracy (%)", showgrid=True, gridcolor="#1e3a5f"),
            legend=dict(bgcolor="#111827"),
            height=350, margin=dict(t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Cross-dataset results ────────────────────────────────────────
    st.markdown("### 🔀 Cross-Dataset Validation (Domain Shift Simulation)")
    cd = cv_data.get("cross_dataset", {})
    if cd:
        cd_rows = []
        for clf_n, scores in cd.items():
            cd_rows.append({
                "Classifier":  clf_n.replace("_cross_dataset", ""),
                "Accuracy":    f"{scores['accuracy']*100:.2f}%",
                "Precision":   f"{scores['precision']*100:.2f}%",
                "Recall":      f"{scores['recall']*100:.2f}%",
                "F1-Score":    f"{scores['f1']*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(cd_rows), use_container_width=True, hide_index=True)

        # Radar comparison
        fig2 = go.Figure()
        cats  = ["Accuracy", "Precision", "Recall", "F1-Score"]
        cols2 = ["#38bdf8", "#e879f9"]
        for i, (clf_n, scores) in enumerate(cd.items()):
            vals = [scores["accuracy"]*100, scores["precision"]*100,
                    scores["recall"]*100, scores["f1"]*100]
            name = clf_n.replace("_cross_dataset", "")
            fig2.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name=name,
                line=dict(color=cols2[i], width=2),
            ))
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#7fa8c9"),
                angularaxis=dict(color="#7fa8c9"),
                bgcolor="#111827",
            ),
            paper_bgcolor="#0d1117",
            font=dict(color="#c9d6e3"),
            height=380, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div style='background:#111827;border:1px solid #1e3a5f;border-radius:10px;
                padding:1rem 1.5rem;margin-top:1rem;'>
      <b style='color:#38bdf8'>ℹ️ Cross-Dataset Validation Methodology</b>
      <p style='color:#c9d6e3;margin:0.5rem 0 0 0;font-size:0.88rem;'>
        In the absence of a second dataset, we simulate domain-shift validation by:
        <br>• <b>Train</b> on the first 80% of shuffled images (Sub-dataset A)
        <br>• <b>Test</b> on the remaining 20% (Sub-dataset B — treated as "unseen domain")
        <br>This rigorously tests generalisation beyond standard K-fold splits.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: About
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📖 About":
    st.markdown('<div class="section-header">📖 About This System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#111827;border:1px solid #1e3a5f;border-radius:12px;
                padding:1.5rem 2rem;'>
      <h3 style='color:#38bdf8;'>Skin Cancer Identification using CNN with PSO-driven Feature
      Selection and ML Classifiers</h3>

      <p><b style='color:#e2e8f0;'>Architecture Overview:</b></p>
      <ul>
        <li><b>Feature Extraction:</b> Pre-trained XceptionNet (2048-dim) + EfficientNetB0 (1280-dim)
            → concatenated 3328-dim feature vector</li>
        <li><b>Feature Scaling:</b> MinMaxScaler normalises to [0,1]</li>
        <li><b>PSO Feature Selection:</b> Particle Swarm Optimisation (pyswarms GlobalBestPSO)
            selects the most discriminative features using accuracy as fitness</li>
        <li><b>ML Classifiers:</b> SVM (RBF kernel), Random Forest (1000 trees),
            KNN, Multi-Layer Perceptron (MLP) Neural Net</li>
        <li><b>Ensemble:</b> Soft-voting ensemble averaging predicted probabilities</li>
        <li><b>Explainability:</b> Grad-CAM heatmaps (Xception), LIME superpixel explanations,
            Random Forest feature importances</li>
        <li><b>Validation:</b> Stratified 5-fold CV + cross-dataset simulation</li>
      </ul>

      <p><b style='color:#e2e8f0;'>Dataset:</b></p>
      <ul>
        <li>HAM10000 (Human Against Machine with 10,000 training images)</li>
        <li>7 classes: akiec, bcc, bkl, df, mel, nv, vasc</li>
        <li>Images: 600×450px dermoscopy, resized to 224×224px for inference</li>
      </ul>

      <p><b style='color:#e2e8f0;'>Tech Stack:</b></p>
      <ul>
        <li>TensorFlow / Keras — CNN feature extraction + Grad-CAM</li>
        <li>scikit-learn — ML classifiers, cross-validation, metrics</li>
        <li>pyswarms — Particle Swarm Optimisation</li>
        <li>LIME — Local Interpretable Model-agnostic Explanations</li>
        <li>Streamlit + Plotly — Interactive web UI</li>
      </ul>

      <p style='color:#7fa8c9;font-size:0.82rem;margin-top:1rem;'>
        ⚠️ <b>Disclaimer:</b> This system is for research and educational purposes only.
        It is not a substitute for professional medical diagnosis.
        Always consult a qualified dermatologist for skin cancer screening.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture flowchart
    st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Image Input",               "HAM10000 dermoscopy images (224×224)"),
        ("2", "CNN Feature Extraction",    "XceptionNet (2048) + EfficientNetB0 (1280)"),
        ("3", "Feature Concatenation",     "3328-dimensional combined vector"),
        ("4", "PSO Feature Selection",     "Optimised feature subset via Particle Swarm"),
        ("5", "ML Classification",         "SVM · Random Forest · KNN · MLP"),
        ("6", "Soft-Voting Ensemble",      "Averaged probability fusion"),
        ("7", "Explainability",            "Grad-CAM · LIME · Feature Importance"),
        ("8", "Cross-Validation",          "5-Fold CV + Cross-Dataset Simulation"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex;align-items:flex-start;margin-bottom:0.6rem;'>
          <div style='min-width:36px;height:36px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                      border-radius:50%;display:flex;align-items:center;justify-content:center;
                      font-weight:700;font-size:0.85rem;color:white;margin-right:1rem;flex-shrink:0;'>
            {num}
          </div>
          <div style='background:#111827;border:1px solid #1e3a5f;border-radius:8px;
                      padding:0.6rem 1rem;flex-grow:1;'>
            <b style='color:#38bdf8;'>{title}</b>
            <div style='color:#94a3b8;font-size:0.82rem;'>{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
