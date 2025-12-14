"""
Dashboard Pemodelan Topik - Aplikasi Utama

Dashboard multi-halaman Streamlit untuk mengeksplorasi hasil pemodelan topik LDA.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """Titik masuk aplikasi utama."""
    st.set_page_config(
        page_title="Dashboard Pemodelan Topik",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Logo di atas navigasi menggunakan st.logo()
    logo_path = Path(__file__).parent.parent / "assets" / "logo.png"
    if logo_path.exists():
        st.logo(str(logo_path), size="large")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown('<p class="main-header">📊 Dashboard Pemodelan Topik</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Eksplorasi topik yang ditemukan dari metadata paper akademik menggunakan LDA</p>',
        unsafe_allow_html=True
    )
    
    # Welcome section
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Selamat Datang!")
        st.markdown("""
        Dashboard ini menyediakan antarmuka interaktif untuk mengeksplorasi hasil 
        pemodelan topik **Latent Dirichlet Allocation (LDA)** pada metadata paper akademik.
        
        **Fitur yang Tersedia:**
        - 📊 **Penjelajah Topik** - Jelajahi dan visualisasikan topik yang ditemukan
        - 📄 **Peramban Dokumen** - Cari dan filter dokumen berdasarkan topik
        - 🔍 **Pencarian Kemiripan** - Temukan dokumen serupa atau analisis teks baru
        - 📈 **Analisis Tren** - Lihat bagaimana topik berubah dari waktu ke waktu
        - 🎯 **Perbandingan Topik** - Bandingkan topik secara berdampingan
        - ⚙️ **Wawasan Model** - Lihat metrik dan konfigurasi model
        
        Gunakan navigasi sidebar untuk mengeksplorasi berbagai fitur.
        """)
    
    with col2:
        st.markdown("### 📚 Statistik Cepat")
        
        # Try to load quick stats
        try:
            from dashboard.utils import load_model, load_data, get_topic_label_manager
            
            model = load_model()
            df = load_data()
            
            if model and df is not None:
                st.metric("Topik", model.model.num_topics)
                st.metric("Dokumen", f"{len(df):,}")
                if model.metadata:
                    st.metric("Koherensi", f"{model.metadata.coherence_score:.4f}")
                
                # Check if labels need to be created
                label_manager = get_topic_label_manager()
                if not label_manager.has_labels():
                    st.markdown("---")
                    st.warning("🏷️ **Label topik belum dikonfigurasi**")
            else:
                st.info("Model belum dimuat. Jalankan notebook terlebih dahulu.")
        except Exception as e:
            st.info("Jalankan Jupyter notebook untuk melatih model terlebih dahulu.")
    
    st.markdown("---")
    
    # Getting started
    st.markdown("### 🚀 Memulai")
    
    st.markdown("""
    Jika Anda belum melatih model, ikuti langkah-langkah berikut:
    
    1. **Instal dependensi**: `uv sync`
    2. **Jalankan notebook** secara berurutan:
       - `01_data_collection.ipynb` - Mengumpulkan data
       - `01b_eda_raw_data.ipynb` - Eksplorasi data mentah
       - `02_data_cleaning.ipynb` - Membersihkan data
       - `02b_eda_clean_data.ipynb` - Eksplorasi data bersih
       - `03_preprocessing.ipynb` - Preprocessing teks
       - `04_lda_modeling.ipynb` - Melatih model LDA
       - `05_analysis_visualization.ipynb` - Membuat visualisasi
    3. **Kembali ke dashboard ini** untuk mengeksplorasi hasil secara interaktif!
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>Dashboard Pemodelan Topik • Dibuat dengan Streamlit</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
