"""
Halaman Pelabelan Topik untuk Dashboard Pemodelan Topik.

Memungkinkan pengguna untuk memberikan label kustom pada setiap topik
untuk memudahkan interpretasi dan navigasi.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dashboard.utils import (
    load_model, check_model_loaded, 
    get_topic_label_manager, TopicLabelManager
)
from dashboard.components.charts import create_wordcloud_figure

st.set_page_config(
    page_title="Pelabelan Topik - Pemodelan Topik",
    page_icon="🏷️",
    layout="wide",
)

st.title("🏷️ Pelabelan Topik")
st.markdown("Berikan label deskriptif untuk setiap topik agar lebih mudah dipahami.")
st.markdown("---")

# Load model
if not check_model_loaded():
    st.stop()

model = load_model()
label_manager = get_topic_label_manager()

num_topics = model.model.num_topics
topics = model.get_topics(num_words=10)

# Get current labels (custom or default)
current_labels = label_manager.get_labels_with_defaults(model)

# Status indicator
col_status1, col_status2 = st.columns([3, 1])
with col_status1:
    if label_manager.has_labels():
        st.success("✅ Label kustom sudah tersimpan")
    else:
        st.info("ℹ️ Menggunakan label otomatis (belum ada label kustom)")

with col_status2:
    if label_manager.has_labels():
        if st.button("🔄 Reset ke Default", type="secondary"):
            if label_manager.reset_labels():
                st.cache_resource.clear()
                st.rerun()

st.markdown("---")

# Instructions
with st.expander("📖 Panduan Pelabelan", expanded=not label_manager.has_labels()):
    st.markdown("""
    **Tips untuk label yang baik:**
    - Gunakan 2-5 kata yang mendeskripsikan tema utama
    - Perhatikan kata-kata teratas untuk memahami konteks topik
    - Contoh: "Manajemen SDM", "Analisis Keuangan", "Sistem Informasi"
    
    **Langkah:**
    1. Lihat kata-kata teratas dan word cloud untuk setiap topik
    2. Ketik label yang sesuai di kolom input
    3. Klik "💾 Simpan Semua Label" di bawah halaman
    """)

# Create form for labeling
st.subheader("📝 Label Topik")

# Store labels in session state for the form
if 'temp_labels' not in st.session_state:
    st.session_state.temp_labels = current_labels.copy()

# Display topics in a grid (2 columns)
for i in range(0, num_topics, 2):
    cols = st.columns(2)
    
    for j, col in enumerate(cols):
        topic_idx = i + j
        if topic_idx >= num_topics:
            break
            
        topic = topics[topic_idx]
        
        with col:
            with st.container(border=True):
                # Topic header
                st.markdown(f"### Topik {topic.topic_id}")
                
                # Top words
                top_words = ", ".join(topic.top_words[:8])
                st.markdown(f"**Kata teratas:** {top_words}")
                
                # Small word cloud
                words = model.get_topic_words(topic.topic_id, num_words=30)
                if words:
                    try:
                        wc_base64 = create_wordcloud_figure(
                            words, 
                            width=300, 
                            height=150,
                            colormap='viridis'
                        )
                        st.image(
                            f"data:image/png;base64,{wc_base64}", 
                            width='stretch'
                        )
                    except Exception:
                        pass
                
                # Label input
                current_label = st.session_state.temp_labels.get(
                    topic.topic_id, 
                    current_labels.get(topic.topic_id, "")
                )
                
                new_label = st.text_input(
                    "Label",
                    value=current_label,
                    key=f"label_input_{topic.topic_id}",
                    placeholder=f"Contoh: {', '.join(topic.top_words[:2])}",
                    label_visibility="collapsed"
                )
                
                # Update temp labels
                st.session_state.temp_labels[topic.topic_id] = new_label

st.markdown("---")

# Save buttons
col_save1, col_save2, col_save3 = st.columns([2, 1, 1])

with col_save1:
    if st.button("💾 Simpan Semua Label", type="primary", width='stretch'):
        # Validate labels
        labels_to_save = {}
        for topic_id, label in st.session_state.temp_labels.items():
            if label.strip():  # Only save non-empty labels
                labels_to_save[topic_id] = label.strip()
        
        if labels_to_save:
            if label_manager.save_labels(labels_to_save):
                st.success("✅ Label berhasil disimpan!")
                st.cache_resource.clear()
                st.balloons()
            else:
                st.error("❌ Gagal menyimpan label. Silakan coba lagi.")
        else:
            st.warning("⚠️ Tidak ada label yang diisi. Silakan isi minimal satu label.")

with col_save2:
    if st.button("🔄 Refresh", width='stretch'):
        st.session_state.temp_labels = label_manager.get_labels_with_defaults(model)
        st.rerun()

with col_save3:
    if st.button("❌ Batal", width='stretch'):
        st.session_state.temp_labels = current_labels.copy()
        st.rerun()

# Preview section
st.markdown("---")
st.subheader("👁️ Preview Label")

preview_data = []
for topic_id in range(num_topics):
    label = st.session_state.temp_labels.get(topic_id, f"Topik {topic_id}")
    preview_data.append({
        "ID": topic_id,
        "Label": label,
        "Kata Teratas": ", ".join(topics[topic_id].top_words[:5])
    })

import pandas as pd
preview_df = pd.DataFrame(preview_data)
st.dataframe(preview_df, width='stretch', hide_index=True)
