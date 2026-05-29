import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import io
from scipy.io.wavfile import write
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Speech Emotion Recognition")

#background website
def set_bg_gradient():
    st.markdown(
        """
        <style>
        /* Mengatur background utama */
        .stApp {
          background: linear-gradient(135deg, #D3CCE3 40%, #E9E4F0 60%);
        }

        /Mengatur tampilan kotak uploader file/
        .stFileUploader {
            background-color: rgba(255, 255, 255, 0.8); /* Putih transparan */
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Opsional: Biar teks di dalam uploader lebih rapi */
        .stFileUploader section {
            color: #4b1248;
        }
        /* Mengatur ukuran teks pada Tab */
        button[data-baseweb="tab"] p {
            font-size: 18px !important; /* Ubah angka ini untuk ukuran lebih besar/kecil */
            font-weight: bold !important;
        }
        /* Mengubah warna teks tab yang sedang aktif/diklik menjadi putih */
        button[data-baseweb="tab"][aria-selected="true"] p {
            color: white !important;
        }

        /* Mengubah warna garis bawah tab yang aktif menjadi putih */
        div[data-baseweb="tab-highlight"] {
            background-color: white !important;
        }

        /* Mengatur tinggi kotak tab agar lebih proporsional */
        button[data-baseweb="tab"] {
            height: 25px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_gradient()

# --- LOAD MODEL & ENCODER ---
@st.cache_resource
def load_model_ser():
    # Menggunakan akun Hugging Face Tiwi yang asli
    hf_username = "rismadwp"
    hf_model_repo = "speech-emotion-recognition" 
    
    # URL Unduhan Langsung (Direct Download) resmi dari Hugging Face
    url_model = f"https://huggingface.co/{hf_username}/{hf_model_repo}/resolve/main/best_model_bi_lstm.keras"
    url_encoder = f"https://huggingface.co/{hf_username}/{hf_model_repo}/resolve/main/label_encoder.pkl"
    
    file_model = "best_model_bi_lstm.keras"
    file_encoder = "label_encoder.pkl"

# --- FUNGSI PRE-PROCESSING ---
def process_audio(y, sr=22050):
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    target_len = int(6.0 * sr)
    if len(y_trim) < target_len:
        y_proc = np.pad(y_trim, (0, target_len - len(y_trim)))
    else:
        y_proc = y_trim[:target_len]
        
    mfcc = librosa.feature.mfcc(y=y_proc, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat = np.concatenate([mfcc, delta, delta2], axis=0).T
    
    if len(feat) < 300:
        feat = np.vstack([feat, np.zeros((300 - len(feat), 120))])
    else:
        feat = feat[:300, :]
    return np.expand_dims(feat, axis=0)

# --- LOGIKA RESET (Session State) ---
# Kita buatkan kunci unik untuk uploader agar bisa dipaksa reset
if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0

def reset_app():
    # Reset untuk Tab 1 (Upload)
    st.session_state['uploader_key'] += 1
    st.session_state['show_result'] = False
    
    # Reset untuk Tab 2 (Rekam)
    st.session_state['show_result_rec'] = False
    if 'audio_rec' in st.session_state:
        del st.session_state['audio_rec']
    
    # Reset Data Hasil (Umum)
    st.session_state['hasil_emosi'] = ""
    st.session_state['hasil_conf'] = 0

# --- TAMPILAN WEBSITE ---
st.markdown("""
    <h1 style='text-align: center; color: #2d2d2d; text-shadow: 2px 2px 5px rgba(255,255,255,0.5);'>
        PENGENALAN EMOSI UCAPAN
    </h1>
    """, unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #2d2d2d; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); '>Website Pendeteksi Emosi Suara menggunakan Arsitektur Bi-LSTM.</p>", unsafe_allow_html=True)
st.divider()

tab1, tab2 = st.tabs(["**Upload File**", "**Rekam Langsung**"])

with tab1:
    # 1. Gunakan key dinamis untuk uploader
    uploader_key = f"uploader_{st.session_state['uploader_key']}"
    uploaded_file = st.file_uploader("**Pilih file audio**", type=["wav"], key=uploader_key)

    if uploaded_file:
        st.audio(uploaded_file)
        
        # Ambil data audio
        audio_bytes = io.BytesIO(uploaded_file.getvalue())
        y_wave, sr_wave = librosa.load(audio_bytes, sr=22050)

        # Tombol Analisis dan Reset
        col1, col2 = st.columns(2)
        with col1:
            btn_analisis = st.button("Analisis Emosi", use_container_width=True)
        with col2:
            st.button("Reset Analisis", use_container_width=True, on_click=reset_app)

        # --- LOGIKA PENYIMPANAN HASIL ---
        # Jika tombol diklik, jalankan analisis dan simpan ke session_state
        if btn_analisis:
            with st.spinner("Sedang menganalisis..."):
                feat = process_audio(y_wave, sr_wave)
                preds = model.predict(feat, verbose=0)
                
                # Simpan ke memori Streamlit biar nggak ilang
                st.session_state['hasil_emosi'] = le.inverse_transform([np.argmax(preds)])[0]
                st.session_state['hasil_conf'] = np.max(preds) * 100
                st.session_state['show_result'] = True

        # --- TAMPILKAN HASIL (Jika sudah pernah dianalisis) ---
        if st.session_state.get('show_result'):
            # Tampilkan Waveform
            st.write("**Visualisasi Bentuk Suara:**")
            fig, ax = plt.subplots(figsize=(10, 2))
            librosa.display.waveshow(y_wave, sr=sr_wave, ax=ax, color='#1f77b4')
            ax.set_axis_off()
            st.pyplot(fig)

            # Tampilkan Tulisan Hasil
            st.success(f"**Hasil Prediksi:** **{st.session_state['hasil_emosi'].upper()}**")
            st.info(f"**Confidence:** **{st.session_state['hasil_conf']:.2f}%**")


with tab2:
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        btn_rekam = st.button("Mulai Rekam", use_container_width=True)
    with col_rec2:
        st.button("Reset Hasil", use_container_width=True, on_click=reset_app)

    if btn_rekam:
        duration = 6.0
        sr = 22050
        with st.spinner("Perekaman berjalan... Silakan bicara!"):
            recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
            sd.wait()
            st.success("Rekaman selesai!")
            
            y_rec = recording.flatten()
            
            # Prediksi
            feat = process_audio(y_rec, sr)
            preds = model.predict(feat, verbose=0)
            
            # Simpan semuanya ke state
            st.session_state['hasil_emosi'] = le.inverse_transform([np.argmax(preds)])[0]
            st.session_state['hasil_conf'] = np.max(preds) * 100
            st.session_state['audio_rec'] = y_rec
            st.session_state['show_result_rec'] = True

    # --- PINTU UTAMA HASIL ---
    if st.session_state.get('show_result_rec'):
        st.divider()
        
        # 1. Audio Player (TAMBAHAN BARU)
        # Bagian ini untuk memutar suara yang baru saja direkam
        st.write("**Putar Ulang Rekaman:**")
        st.audio(st.session_state['audio_rec'], sample_rate=22050)
        
        # 2. Visualisasi Waveform
        st.write("**Visualisasi Bentuk Suara Rekaman:**")
        fig_rec, ax_rec = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(st.session_state['audio_rec'], sr=22050, ax=ax_rec, color='#1f77b4')
        ax_rec.set_axis_off()
        st.pyplot(fig_rec)

        # 3. Metrik Hasil
        st.success(f"**Hasil Prediksi:** **{st.session_state['hasil_emosi'].upper()}**")
        st.info(f"**Confidence:** **{st.session_state['hasil_conf']:.2f}%**")
