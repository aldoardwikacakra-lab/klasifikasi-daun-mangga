import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import time

st.set_page_config(
    page_title="Klasifikasi Penyakit Daun Mangga",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Background ── */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 50%, #f0fdf4 100%);
}
[data-testid="stHeader"] { background: transparent; }

/* ── Semua teks di area utama → hitam gelap ── */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] div,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stMarkdownContainer"] *,
.stMarkdown * { color: #111827 !important; }

/* ── Metric box ── */
[data-testid="metric-container"] {
    background: white !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    border: 1px solid #dcfce7 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.08) !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #111827 !important;
    font-weight: 700 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] label { color: #111827 !important; font-weight: 500 !important; }

/* Teks di dalam dropzone (kotak gelap) → putih agar terbaca */
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] * { color: #ffffff !important; font-weight: 500 !important; }

/* Tombol Browse di dalam dropzone */
[data-testid="stFileUploaderDropzone"] button {
    color: #ffffff !important;
    border-color: rgba(255,255,255,0.5) !important;
    font-weight: 600 !important;
}

/* ── Success / info / alert ── */
[data-testid="stAlert"] *,
.stSuccess *, .stInfo * { color: #111827 !important; }

/* ── Spinner text ── */
.stSpinner * { color: #111827 !important; }

/* ── Caption gambar ── */
.stImage figcaption, .stImage caption { color: #374151 !important; }

/* ── Heading markdown ── */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #111827 !important;
    font-weight: 700 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14532d 0%, #166534 60%, #15803d 100%) !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] code {
    color: #ffffff !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.25) !important; }

/* ── Custom components ── */
.main-header {
    background: linear-gradient(135deg, #14532d 0%, #16a34a 50%, #22c55e 100%);
    padding: 2.5rem 2rem; border-radius: 16px; text-align: center;
    margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(20,83,45,0.3);
}
.main-header h1 { color:#fff !important; font-size:2rem; font-weight:700; margin:0 0 .5rem; }
.main-header p  { color:#bbf7d0 !important; font-size:1rem; margin:0; }

.card { background:white; border-radius:12px; padding:1.5rem;
        box-shadow:0 4px 16px rgba(0,0,0,.08); border:1px solid #dcfce7; margin-bottom:1rem; }
.card-title { font-size:1rem; font-weight:700; color:#14532d !important; margin-bottom:.75rem; }

.result-box { border-radius:16px; padding:2rem; text-align:center;
              margin:1rem 0; box-shadow:0 8px 24px rgba(0,0,0,.2); }
.result-box * { color: #ffffff !important; }
.result-box-healthy { background: linear-gradient(135deg,#064e3b,#059669); }
.result-box-disease  { background: linear-gradient(135deg,#7f1d1d,#dc2626); }
.result-label { font-size:1rem; opacity:.85; margin-bottom:.25rem; }
.result-class { font-size:2rem; font-weight:700; margin:.25rem 0; }
.result-conf  { font-size:1.1rem; opacity:.9; }

.prob-bar-wrap { margin:.4rem 0; }
.prob-label { display:flex; justify-content:space-between; font-size:.85rem;
              margin-bottom:.2rem; font-weight:600; }
.prob-label * { color:#111827 !important; }
.prob-bar-bg   { background:#d1fae5; border-radius:99px; height:12px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:99px; }

.pill { display:inline-block; padding:.25rem .75rem; border-radius:99px;
        font-size:.78rem; font-weight:700; margin:.2rem .1rem; }
.pill-green { background:#dcfce7; color:#14532d !important; }
.pill-red   { background:#fee2e2; color:#991b1b !important; }
.pill-blue  { background:#dbeafe; color:#1e40af !important; }

footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# Constants
IMG_SIZE   = (300, 300)
MODEL_PATH = "mango_efficientnetb7_final (2).keras"

CLASS_NAMES = [
    "Anthracnose",       # 1
    "Bacterial Canker",  # 2
    "Cutting Weevil",    # 3
    "Die Back",          # 4
    "Gall Midge",        # 5
    "Healthy",           # 6
    "Powdery Mildew",    # 7
    "Sooty Mould",       # 8
]

CLASS_INFO = {
    "Anthracnose":    {"id":"Anthracnose",    "status":"disease","emoji":"🍂",
        "desc":"Munculnya bercak berwarna coklat kehitaman pada permukaan daun serta area jaringan yang mengalami nekrosis.",
        "penanganan":"Semprot fungisida mankozeb atau tembaga oksiklorida. Pangkas bagian terinfeksi.",
        "severity":"Sedang"},
    "Bacterial Canker":{"id":"Bacterial Canker","status":"disease","emoji":"🔴",
        "desc":"Adanya luka atau bercak basah berwarna coklat kekuningan pada daun yang disebabkan oleh infeksi bakteri.",
        "penanganan":"Gunakan bakterisida berbahan tembaga. Sterilkan alat pemangkas.",
        "severity":"Tinggi"},
    "Cutting Weevil": {"id":"Cutting Weevil", "status":"disease","emoji":"🐛",
        "desc":"Hama yang menyebabkan kerusakan berupa potongan atau lubang tidak beraturan pada bagian daun akibat aktivitas makan serangga.",
        "penanganan":"Aplikasikan insektisida sistemik. Musnahkan bagian tanaman yang rusak.",
        "severity":"Sedang"},
    "Die Back":       {"id":"Die Back",        "status":"disease","emoji":"🍁",
        "desc":"Mengeringnya ujung daun atau cabang secara bertahap hingga jaringan tanaman mengalami kematian.",
        "penanganan":"Pangkas bagian terinfeksi hingga jaringan sehat. Oleskan fungisida.",
        "severity":"Tinggi"},
    "Gall Midge":     {"id":"Gall Midge",      "status":"disease","emoji":"🦟",
        "desc":"Terbentuknya benjolan atau pembengkakan abnormal pada permukaan daun akibat serangan larva serangga.",
        "penanganan":"Semprot insektisida saat flush tunas baru. Musnahkan daun bergall.",
        "severity":"Sedang"},
    "Healthy":        {"id":"Healthy",          "status":"healthy","emoji":"✅",
        "desc":"Daun mangga dalam kondisi sehat tanpa gejala penyakit atau serangan hama.",
        "penanganan":"Lanjutkan perawatan rutin: pemupukan, penyiraman, dan pemangkasan berkala.",
        "severity":"Tidak ada"},
    "Powdery Mildew": {"id":"Powdery Mildew",  "status":"disease","emoji":"⬜",
        "desc":"Munculnya lapisan putih menyerupai tepung pada permukaan daun akibat infeksi jamur.",
        "penanganan":"Semprotkan sulfur atau fungisida sistemik. Pastikan sirkulasi udara baik.",
        "severity":"Sedang"},
    "Sooty Mould":    {"id":"Sooty Mould",      "status":"disease","emoji":"⬛",
        "desc":"Jamur yang tumbuh di atas embun madu serangga sehingga menutupi daun dengan lapisan hitam.",
        "penanganan":"Kendalikan serangga (kutu daun). Bersihkan daun dengan air sabun.",
        "severity":"Rendah"},
}

SEVERITY_COLOR = {
    "Tidak ada":"#16a34a","Rendah":"#ca8a04","Sedang":"#ea580c","Tinggi":"#dc2626"
}

# Load Model
@st.cache_resource(show_spinner=False)
def load_model():
    import zipfile, tempfile, shutil, os
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        pass
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(MODEL_PATH, 'r') as z:
        z.extractall(tmpdir)
    cfg_path = os.path.join(tmpdir, "config.json")
    with open(cfg_path, "r") as f:
        cfg_str = f.read()
    cfg_str = cfg_str.replace('"quantization_config": null,', '')
    cfg_str = cfg_str.replace(', "quantization_config": null', '')
    with open(cfg_path, "w") as f:
        f.write(cfg_str)
    patched = os.path.join(tmpdir, "patched.keras")
    files_in_zip = [f for f in ["metadata.json","config.json","model.weights.h5"]
                    if os.path.exists(os.path.join(tmpdir, f))]
    with zipfile.ZipFile(patched, 'w') as z:
        for fname in files_in_zip:
            z.write(os.path.join(tmpdir, fname), fname)
    model = tf.keras.models.load_model(patched)
    shutil.rmtree(tmpdir)
    return model

# Preprocess & Predict
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict(model, image: Image.Image):
    probs = model.predict(preprocess(image), verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

def prob_bar(label, prob, is_top=False):
    pct   = prob * 100
    color = "#16a34a" if is_top else "#6ee7b7"
    return f"""
    <div class="prob-bar-wrap">
      <div class="prob-label">
        <span style="color:#111827;font-weight:600;">{label}</span>
        <span style="color:#111827;font-weight:700;">{pct:.1f}%</span>
      </div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:{pct}%;background:{color};"></div>
      </div>
    </div>"""

# SIDEBAR
with st.sidebar:
    st.markdown("## 🥭 Klasifikasi Penyakit Daun Mangga")
    st.markdown("""

---
**Spesifikasi Model:**
""")
    st.markdown("🧠 **Arsitektur:** EfficientNetB7")
    st.markdown("📐 **Input Size:** 300 × 300 px")
    st.markdown("⚙️ **Preprocessing:** EfficientNet preprocess_input")
    st.markdown("🏷️ **Jumlah Kelas:** 8 kelas")
    st.markdown("🎯 **Test Accuracy:** ~99%")
    st.markdown("---")
    st.markdown("**🌿 8 Kelas Penyakit:**")
    for i, name in enumerate(CLASS_NAMES):
        info = CLASS_INFO[name]
        st.markdown(f"{i+1}. {info['emoji']} {info['id']}")
    st.markdown("---")
    st.markdown("**📋 Cara Penggunaan:**")
    st.markdown("1. Upload foto daun mangga\n2. Tunggu hasil prediksi\n3. Baca rekomendasi penanganan")

# MAIN
st.markdown("""
<div class="main-header">
  <h1>🥭 Klasifikasi Penyakit Daun Mangga</h1>
  <p>Convolutional Neural Network (CNN) - EfficientNetB7</p>
</div>
""", unsafe_allow_html=True)

# Metric row
c1, c2, c3, c4 = st.columns(4)
for col, label, val in [
    (c1, "🧠 Arsitektur",  "EfficientNetB7"),
    (c2, "📐 Input Size",  "300 × 300 px"),
    (c3, "🏷️ Jumlah Kelas","8 Kelas"),
    (c4, "🎯 Test Accuracy","~99%"),
]:
    col.markdown(f"""
    <div style="background:white;border-radius:10px;padding:1rem;
                border:1px solid #dcfce7;box-shadow:0 2px 8px rgba(0,0,0,.08);
                text-align:center;">
        <div style="font-size:.8rem;color:#374151;font-weight:600;margin-bottom:.3rem;">{label}</div>
        <div style="font-size:1.4rem;color:#111827;font-weight:800;">{val}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.spinner("⏳ Memuat model EfficientNetB7..."):
    model = load_model()
st.success("✅ Model berhasil dimuat!")

st.markdown("<br>", unsafe_allow_html=True)

# Upload zone
st.markdown('<div class="card"><div class="card-title">📤 Upload Citra Daun Mangga</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Pilih gambar daun mangga (JPG / JPEG / PNG)",
    type=["jpg","jpeg","png"],
    label_visibility="visible"
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    image = Image.open(uploaded)
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="card"><div class="card-title">🖼️ Citra yang Diupload</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption=f"📁 {uploaded.name}")
        w, h = image.size
        st.markdown(
            f'<span class="pill pill-blue">📏 {w}×{h} px</span>'
            f'<span class="pill pill-blue">📦 {uploaded.size/1024:.1f} KB</span>',
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        with st.spinner("🔍 Menganalisis citra..."):
            t0 = time.time()
            cls, conf, probs = predict(model, image)
            elapsed = time.time() - t0

        info    = CLASS_INFO[cls]
        is_h    = info["status"] == "healthy"
        box_cls = "result-box-healthy" if is_h else "result-box-disease"

        st.markdown(f"""
        <div class="result-box {box_cls}">
          <div class="result-label">Hasil Prediksi</div>
          <div class="result-class">{info['emoji']} {info['id']}</div>
          <div class="result-conf">Kepercayaan: <b>{conf*100:.2f}%</b></div>
          <div style="font-size:.8rem;opacity:.8;margin-top:.5rem;">⏱ {elapsed:.2f} detik</div>
        </div>
        """, unsafe_allow_html=True)

        sev_color = SEVERITY_COLOR.get(info["severity"], "#6b7280")
        st.markdown(f"""
        <span class="pill {'pill-green' if is_h else 'pill-red'}">
          {'✅ Daun Sehat' if is_h else '⚠️ Terinfeksi'}
        </span>
        <span class="pill pill-blue">🔬 {cls}</span>
        <span style="display:inline-block;padding:.25rem .75rem;border-radius:99px;
          font-size:.78rem;font-weight:700;background:{sev_color}22;color:{sev_color};">
          ⚡ {info['severity']}
        </span>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probabilitas semua kelas
    st.markdown('<div class="card"><div class="card-title">📊 Probabilitas Semua Kelas</div>', unsafe_allow_html=True)
    bars = ""
    for i in np.argsort(probs)[::-1]:
        inf  = CLASS_INFO[CLASS_NAMES[i]]
        bars += prob_bar(
            f"{inf['emoji']} {CLASS_NAMES[i]}",
            float(probs[i]),
            is_top=(i == int(np.argmax(probs)))
        )
    st.markdown(bars, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detail penyakit
    sev_color = SEVERITY_COLOR.get(info["severity"], "#6b7280")
    st.markdown(f"""
    <div class="card">
      <div class="card-title">🔬 Detail: {info['emoji']} {info['id']}</div>
      <p style="color:#111827;font-size:.95rem;margin-bottom:1rem;">{info['desc']}</p>
      <div style="background:#f0fdf4;border-left:4px solid #16a34a;
                  padding:.75rem 1rem;border-radius:0 8px 8px 0;">
        <b style="color:#14532d;">💊 Rekomendasi Penanganan:</b><br>
        <span style="color:#111827;">{info['penanganan']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card" style="text-align:center;padding:3rem;border:2px dashed #6ee7b7;">
      <div style="font-size:3rem;margin-bottom:1rem;">📸</div>
      <p style="font-size:1.1rem;color:#14532d;font-weight:700;">
        Upload citra daun mangga untuk memulai klasifikasi
      </p>
      <p style="color:#374151;font-size:.9rem;">Mendukung format JPG, JPEG, dan PNG</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌿 Kelas yang Dapat Dideteksi")
    cols = st.columns(4)
    for i, (name, info) in enumerate(CLASS_INFO.items()):
        with cols[i % 4]:
            border = "#16a34a" if info["status"]=="healthy" else "#dc2626"
            bg     = "#f0fdf4" if info["status"]=="healthy" else "#fff1f2"
            tc     = "#14532d" if info["status"]=="healthy" else "#991b1b"
            st.markdown(f"""
            <div style="background:{bg};border:1.5px solid {border};border-radius:10px;
                        padding:1rem;text-align:center;margin-bottom:.75rem;">
              <div style="font-size:1.8rem;">{info['emoji']}</div>
              <div style="font-weight:700;color:#111827;font-size:.9rem;margin-top:.3rem;">
                {info['id']}
              </div>
              <div style="color:#374151;font-size:.75rem;font-weight:500;margin-top:.2rem;">
                Kelas {i+1}
              </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#374151;font-size:.85rem;font-weight:500;
            padding:1rem;border-top:1px solid #bbf7d0;margin-top:2rem;">
  🥭 Klasifikasi Penyakit Daun Mangga · CNN · EfficientNetB7
</div>
""", unsafe_allow_html=True)