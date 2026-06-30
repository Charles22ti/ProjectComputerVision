import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import datetime
import io
import base64
import sqlite3

st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🤖",
    layout="centered"
)

def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama_file TEXT,
            waktu TEXT,
            hasil TEXT,
            score TEXT,
            image_data TEXT,
            is_ai INTEGER
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(nama_file, waktu, hasil, score, image_data, is_ai):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO scan_history (nama_file, waktu, hasil, score, image_data, is_ai)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (nama_file, waktu, hasil, score, image_data, 1 if is_ai else 0))
    conn.commit()
    conn.close()

def load_from_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT nama_file, waktu, hasil, score, image_data, is_ai FROM scan_history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    history_list = []
    for row in rows:
        history_list.append({
            "nama_file": row[0],
            "waktu": row[1],
            "hasil": row[2],
            "score": row[3],
            "image_data": row[4],
            "is_ai": bool(row[5])
        })
    return history_list

def clear_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scan_history")
    conn.commit()
    conn.close()

init_db()

def image_to_base64(img):
    buffered = io.BytesIO()
    img_thumbnail = img.copy()
    img_thumbnail.thumbnail((120, 120))
    img_thumbnail.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

st.markdown("""
    <style>
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .stFileUploader section {
        background-color: #111827 !important;
        border: 1px dashed #1E293B !important;
        border-radius: 12px !important;
    }
    .stFileUploader label p { color: #94A3B8 !important; }

    /* Menyembunyikan seluruh progress bar bawaan Streamlit secara paksa */
    .stProgress, div[data-testid="stProgress"] { display: none !important; }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; background-color: #111827; padding: 6px; border-radius: 10px; border: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px; background-color: transparent; border-radius: 8px; color: #64748B;
        font-weight: 500; font-size: 14px; padding: 8px 16px; transition: all 0.2s ease; border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #F1F5F9; background-color: #1E293B; }
    .stTabs [aria-selected="true"] { color: #06B6D4 !important; background-color: #1E293B !important; font-weight: 600; }

    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #0891B2 0%, #0284C7 100%); color: white; font-weight: 600;
        border: none; padding: 12px 24px; width: 100%; border-radius: 10px;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.2); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div.stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 0 25px rgba(6, 182, 212, 0.45);
        background: linear-gradient(135deg, #06B6D4 0%, #0369A1 100%); color: white;
    }

    /* Result Box Container */
    .result-box {
        padding: 24px; border-radius: 14px; margin-top: 25px; margin-bottom: 25px;
        background: rgba(255, 255, 255, 0.02); text-align: center;
    }
    .result-box-title { font-size: 12px; font-weight: 600; color: #64748B; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
    .result-box-label { font-size: 32px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 8px; }
    .result-box-score { color: #94A3B8; font-size: 14px; margin-bottom: 18px; }

    /* Progress Bar Styles */
    .progress-track { width: 100%; background-color: #1E293B; height: 24px; border-radius: 12px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 12px 0 0 12px; }

    /* Dashboard & Cards */
    .stat-card { background-color: #111827; border: 1px solid #1E293B; padding: 16px; border-radius: 10px; text-align: center; }
    .history-card { padding: 16px; border-radius: 12px; background-color: #111827; margin-bottom: 14px; border: 1px solid #1E293B; display: flex; align-items: center; gap: 16px; }
    .history-card:hover { border: 1px solid #334155; }
    .thumb-container { width: 60px; height: 60px; border-radius: 8px; overflow: hidden; border: 1px solid #334155; flex-shrink: 0; }
    .thumb-img { width: 100%; height: 100%; object-fit: cover; }
    .history-info { flex-grow: 1; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "./my_vit_model"
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    return processor, model

try:
    processor, model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model ViT. Error: {e}")
    st.stop()

# --- HEADER ---
st.markdown("<h2 style='text-align: center; color: #F8FAFC; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 0px;'>Image Authenticator</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B; font-size: 14px; margin-bottom: 35px;'>Vision Transformer (ViT) Deep-Analysis Pipeline</p>", unsafe_allow_html=True)

tab_deteksi, tab_riwayat = st.tabs(["🔍 Deteksi Gambar", "📜 Riwayat Deteksi"])

if 'active_result' not in st.session_state:
    st.session_state.active_result = None

# ==================== TAB 1: DETEKSI GAMBAR ====================
with tab_deteksi:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is None:
        st.session_state.active_result = None
        
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("<div style='margin: 15px 0;'>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Mulai Deteksi"):
            with st.spinner("Mengekstraksi representasi patch gambar..."):
                inputs = processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                
                labels = model.config.id2label
                raw_label = labels[predicted_class_idx]
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence = probs[0][predicted_class_idx].item()
                confidence_pct = confidence * 100
                
                is_ai = "ai" in raw_label.lower()
                if is_ai:
                    display_label = "Buatan AI"
                    accent_color = "#EF4444"  # Merah
                else:
                    display_label = "Buatan Manusia"
                    accent_color = "#10B981"  # Hijau
                
                st.session_state.active_result = {
                    "label": display_label,
                    "color": accent_color,
                    "percentage": confidence_pct
                }
                
                img_b64 = image_to_base64(image)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                score_str = f"{confidence_pct:.2f}%"
                save_to_db(uploaded_file.name, timestamp, display_label, score_str, img_b64, is_ai)

        # RENDER BLOK HASIL KUSTOM (Sangat Bersih dari Konflik Tag HTML)
        if st.session_state.active_result is not None:
            res = st.session_state.active_result
            
            # Pisahkan pembuatan string persentase untuk mencegah gangguan pembacaan bracket `{}` di CSS inline
            width_style = f"width: {res['percentage']:.2f}%; background-color: {res['color']};"
            border_style = f"border: 1px solid {res['color']}55; box-shadow: 0 0 20px {res['color']}1A;"
            color_style = f"color: {res['color']};"
            
            st.markdown(f"""
                <div class="result-box" style="{border_style}">
                    <div class="result-box-title">Klasifikasi Terdeteksi</div>
                    <div class="result-box-label" style="{color_style}">{res['label']}</div>
                    <div class="result-box-score">Tingkat Keyakinan: <span style="color: #F1F5F9; font-weight: 600;">{res['percentage']:.2f}%</span></div>
                    <div class="progress-track">
                        <div class="progress-fill" style="{width_style}"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

with tab_riwayat:
    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
    
    db_history = load_from_db()
    
    if not db_history:
        st.markdown("<p style='color: #64748B; text-align: center; font-size: 14px;'>Belum ada riwayat pemindaian gambar permanen yang tersimpan.</p>", unsafe_allow_html=True)
    else:
        total_checks = len(db_history)
        ai_count = sum(1 for item in db_history if item["is_ai"])
        real_count = total_checks - ai_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="stat-card"><div style="font-size:11px; color:#64748B; text-transform:uppercase; font-weight:600;">Total Scan</div><div style="font-size:22px; font-weight:700; color:#06B6D4;">{total_checks}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div style="font-size:11px; color:#64748B; text-transform:uppercase; font-weight:600;">Terdeteksi AI</div><div style="font-size:22px; font-weight:700; color:#EF4444;">{ai_count}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="stat-card"><div style="font-size:11px; color:#64748B; text-transform:uppercase; font-weight:600;">Buatan Manusia</div><div style="font-size:22px; font-weight:700; color:#10B981;">{real_count}</div></div>', unsafe_allow_html=True)
            
        st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
        
        for item in db_history:
            status_color = "#EF4444" if item["is_ai"] else "#10B981"
            
            st.markdown(f"""
                <div class="history-card">
                    <div class="thumb-container">
                        <img class="thumb-img" src="data:image/jpeg;base64,{item['image_data']}" />
                    </div>
                    <div class="history-info">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600; color: #F1F5F9; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 220px;">{item['nama_file']}</span>
                            <span style="color: {status_color}; font-weight: 700; font-size: 13px; letter-spacing: 0.5px;">{item['hasil']}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                            <span style="color: #64748B; font-size: 11px;">🕒 {item['waktu']}</span>
                            <span style="color: #94A3B8; font-size: 12px;">Akurasi: {item['score']}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        if st.button("Kosongkan Semua Riwayat"):
            clear_db()
            st.session_state.active_result = None
            st.rerun()