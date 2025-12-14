"""
Halaman Penjelajah Topik untuk Dashboard Pemodelan Topik.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dashboard.utils import (
    load_model, load_data, load_processed_docs,
    get_analyzer, get_visualizer, check_model_loaded,
    get_topic_label_manager
)
from dashboard.components.charts import (
    create_wordcloud_figure, create_topic_bar_chart,
    create_topic_prevalence_chart
)
from dashboard.components.filters import topic_selector

st.set_page_config(
    page_title="Penjelajah Topik - Pemodelan Topik",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Penjelajah Topik")
st.markdown("Eksplorasi topik yang ditemukan, lihat word cloud, dan analisis konten topik.")
st.markdown("---")

# Load model
if not check_model_loaded():
    st.stop()

model = load_model()
df = load_data()
analyzer = get_analyzer(model)
visualizer = get_visualizer(model)

num_topics = model.model.num_topics

# Load topic labels
label_manager = get_topic_label_manager()
topic_labels = label_manager.get_labels_with_defaults(model)

# Sidebar
with st.sidebar:
    st.header("Opsi")
    
    selected_topic = topic_selector(
        num_topics,
        key="explorer_topic",
        label="Pilih Topik untuk Dijelajahi",
        include_all=False,
        labels=topic_labels,
    )
    
    num_words = st.slider(
        "Jumlah Kata",
        min_value=5,
        max_value=30,
        value=15,
        key="num_words",
    )

# Main content
tab1, tab2, tab3 = st.tabs(["📈 Ringkasan Topik", "☁️ Word Cloud", "🔤 Detail Kata"])

with tab1:
    st.subheader("Prevalensi Topik")
    
    if analyzer:
        try:
            prevalence = analyzer.compute_topic_prevalence()
            fig = create_topic_prevalence_chart(prevalence)
            st.plotly_chart(fig, width='stretch')
            
            # Summary stats
            st.markdown("### Ringkasan Topik")
            
            cols = st.columns(4)
            for i, row in prevalence.iterrows():
                col_idx = i % 4
                with cols[col_idx]:
                    topic_id = int(row['topic_id'])
                    mean_prob = row['mean_probability']
                    num_dominant = int(row['num_dominant'])
                    
                    st.metric(
                        f"Topik {topic_id}",
                        f"{mean_prob:.3f}",
                        f"{num_dominant} dok",
                    )
        except Exception as e:
            st.error(f"Error menghitung prevalensi: {e}")
    
    # All topics list
    st.markdown("### Semua Topik")
    
    topics = model.get_topics(num_words=8)
    for topic in topics:
        label = topic_labels.get(topic.topic_id, f"Topik {topic.topic_id}")
        with st.expander(f"Topik {topic.topic_id}: {label}"):
            st.write(f"**Kata teratas**: {', '.join(topic.top_words)}")

with tab2:
    st.subheader(f"Word Cloud - Topik {selected_topic}")
    
    # Get word weights
    words = model.get_topic_words(selected_topic, num_words=50)
    
    if words:
        # Create word cloud
        wc_base64 = create_wordcloud_figure(words)
        st.image(f"data:image/png;base64,{wc_base64}", width='stretch')
    else:
        st.info("Tidak ada kata yang ditemukan untuk topik ini")
    
    # Color options
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Ringkasan Topik {selected_topic}**")
        topic_info = topics[selected_topic]
        st.write(f"Kata teratas: {', '.join(topic_info.top_words[:10])}")
    
    with col2:
        if analyzer and df is not None:
            try:
                prevalence = analyzer.compute_topic_prevalence()
                topic_prev = prevalence[prevalence['topic_id'] == selected_topic].iloc[0]
                st.metric("Probabilitas Rata-rata", f"{topic_prev['mean_probability']:.4f}")
                st.metric("Dokumen (Dominan)", int(topic_prev['num_dominant']))
            except:
                pass

with tab3:
    st.subheader(f"Detail Kata - Topik {selected_topic}")
    
    # Get words with weights
    words = model.get_topic_words(selected_topic, num_words=num_words)
    
    if words:
        # Bar chart
        word_list = [w for w, _ in words]
        weight_list = [w for _, w in words]
        
        fig = create_topic_bar_chart(
            word_list, weight_list, 
            selected_topic,
            height=max(400, len(words) * 25)
        )
        st.plotly_chart(fig, width='stretch')
        
        # Table
        st.markdown("### Tabel Bobot Kata")
        
        import pandas as pd
        word_df = pd.DataFrame({
            "Peringkat": range(1, len(words) + 1),
            "Kata": word_list,
            "Bobot": weight_list,
        })
        st.dataframe(word_df, width='stretch', hide_index=True)
    else:
        st.info("Tidak ada kata yang ditemukan untuk topik ini")

# pyLDAvis section
st.markdown("---")
st.subheader("📊 Visualisasi Topik Interaktif (pyLDAvis)")

try:
    from src.config import get_settings
    settings = get_settings()
    pyldavis_path = settings.outputs_dir / "pyldavis.html"
    
    if pyldavis_path.exists():
        with open(pyldavis_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.info("Visualisasi pyLDAvis tidak ditemukan. Jalankan notebook 05_analysis_visualization untuk membuatnya.")
except Exception as e:
    st.warning(f"Tidak dapat memuat pyLDAvis: {e}")
