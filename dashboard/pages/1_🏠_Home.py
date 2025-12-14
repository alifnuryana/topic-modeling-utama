"""
Halaman Beranda untuk Dashboard Pemodelan Topik.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dashboard.utils import load_model, load_data, check_model_loaded, get_topic_label_manager, setup_logo

st.set_page_config(
    page_title="Beranda - Pemodelan Topik",
    page_icon="🏠",
    layout="wide",
)

setup_logo()

st.title("🏠 Beranda")
st.markdown("---")

# Check if model is loaded
model = load_model()
df = load_data()

if model is None:
    st.warning("⚠️ Model belum dimuat. Silakan jalankan notebook terlebih dahulu untuk melatih model.")
    
    st.markdown("""
    ### Memulai
    
    Ikuti langkah-langkah berikut untuk melatih model topik:
    
    1. **Instal dependensi**: Jalankan `uv sync` di terminal
    2. **Jalankan notebook** secara berurutan:
       - `01_data_collection.ipynb` - Mengumpulkan data dari repositori
       - `01b_eda_raw_data.ipynb` - Eksplorasi data mentah
       - `02_data_cleaning.ipynb` - Membersihkan data
       - `02b_eda_clean_data.ipynb` - Eksplorasi data bersih
       - `03_preprocessing.ipynb` - Preprocessing teks
       - `04_lda_modeling.ipynb` - Melatih model LDA
       - `05_analysis_visualization.ipynb` - Membuat visualisasi
    3. **Kembali ke sini** untuk mengeksplorasi hasil!
    """)
else:
    # Load label manager
    label_manager = get_topic_label_manager()
    topic_labels = label_manager.get_labels_with_defaults(model)
    
    # Display model info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Topik", model.model.num_topics)
    
    with col2:
        if df is not None:
            st.metric("📚 Dokumen", f"{len(df):,}")
    
    with col3:
        if model.metadata:
            st.metric("📈 Koherensi", f"{model.metadata.coherence_score:.4f}")
    
    with col4:
        if model.metadata:
            st.metric("📝 Kosakata", f"{model.metadata.vocabulary_size:,}")
    
    st.markdown("---")
    
    # Labeling wizard prompt (if no labels yet)
    if not label_manager.has_labels():
        with st.container(border=True):
            st.markdown("### 🏷️ Langkah Selanjutnya: Labeli Topik Anda")
            st.markdown(f"""
            Model Anda memiliki **{model.model.num_topics} topik** yang belum diberi label kustom. 
            Memberikan label yang deskriptif akan membantu memahami setiap topik dengan lebih mudah.
            """)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.page_link("pages/8_🏷️_Topic_Labeling.py", label="🏷️ Label Topik Sekarang", icon="✏️")
            with col_btn2:
                st.caption("Anda juga bisa melakukan ini nanti dari menu sidebar.")
        
        st.markdown("---")
    
    # Topics overview
    st.subheader("📋 Ringkasan Topik")
    
    topics = model.get_topics(num_words=8)
    
    # Display topics in a grid
    cols = st.columns(2)
    
    for i, topic in enumerate(topics):
        with cols[i % 2]:
            label = topic_labels.get(topic.topic_id, "")
            words = ", ".join(topic.top_words[:6])
            if label and label != words[:len(label)]:
                st.markdown(f"**Topik {topic.topic_id}: {label}**")
                st.caption(f"Kata: {words}")
            else:
                st.markdown(f"**Topik {topic.topic_id}**: {words}")
    
    st.markdown("---")
    
    # Quick navigation
    st.subheader("🚀 Navigasi Cepat")
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        st.markdown("""
        **📊 Penjelajah Topik**
        
        Jelajahi topik, lihat word cloud, dan eksplorasi kata-kata teratas.
        """)
    
    with nav_col2:
        st.markdown("""
        **📄 Peramban Dokumen**
        
        Cari dan filter dokumen berdasarkan topik atau kata kunci.
        """)
    
    with nav_col3:
        st.markdown("""
        **🏷️ Pelabelan Topik**
        
        Berikan label kustom untuk setiap topik.
        """)
    
    # Dataset info
    if df is not None:
        st.markdown("---")
        st.subheader("📊 Ringkasan Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'year' in df.columns:
                year_min = int(df['year'].min()) if df['year'].notna().any() else "N/A"
                year_max = int(df['year'].max()) if df['year'].notna().any() else "N/A"
                st.write(f"**Rentang Tahun**: {year_min} - {year_max}")
            
            if 'title' in df.columns:
                avg_title_len = df['title'].str.len().mean()
                st.write(f"**Rata-rata Panjang Judul**: {avg_title_len:.0f} karakter")
        
        with col2:
            if 'dominant_topic' in df.columns:
                most_common_topic = df['dominant_topic'].mode().iloc[0]
                topic_label = topic_labels.get(int(most_common_topic), f"Topik {int(most_common_topic)}")
                st.write(f"**Topik Paling Umum**: {topic_label}")
            
            if 'abstract' in df.columns:
                avg_abstract_len = df['abstract'].str.split().str.len().mean()
                st.write(f"**Rata-rata Panjang Abstrak**: {avg_abstract_len:.0f} kata")
