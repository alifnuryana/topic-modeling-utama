"""
Halaman Perbandingan Topik untuk Dashboard Pemodelan Topik.
"""

import streamlit as st
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dashboard.utils import load_model, load_data, get_analyzer, get_visualizer, check_model_loaded, get_topic_label_manager
from dashboard.components.charts import (
    create_topic_comparison_chart, create_wordcloud_figure, create_heatmap
)
from dashboard.components.filters import topic_selector

st.set_page_config(
    page_title="Perbandingan Topik - Pemodelan Topik",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Perbandingan Topik")
st.markdown("Bandingkan topik secara berdampingan untuk memahami persamaan dan perbedaannya.")
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
    st.header("Pilih Topik untuk Dibandingkan")
    
    topic_a = st.selectbox(
        "Topik A",
        options=list(range(num_topics)),
        index=0,
        format_func=lambda x: f"Topik {x}: {topic_labels.get(x, '')}",
        key="compare_topic_a",
    )
    
    topic_b = st.selectbox(
        "Topik B",
        options=list(range(num_topics)),
        index=min(1, num_topics - 1),
        format_func=lambda x: f"Topik {x}: {topic_labels.get(x, '')}",
        key="compare_topic_b",
    )
    
    st.markdown("---")
    
    num_words = st.slider(
        "Jumlah Kata",
        min_value=5,
        max_value=20,
        value=10,
        key="compare_num_words",
    )

# Get topic data
topic_a_words = model.get_topic_words(topic_a, num_words)
topic_b_words = model.get_topic_words(topic_b, num_words)

# Main comparison
tab1, tab2, tab3 = st.tabs(["📊 Perbandingan Kata", "☁️ Word Cloud", "🔗 Analisis Tumpang Tindih"])

with tab1:
    st.subheader("Perbandingan Berdampingan")
    
    # Comparison chart
    fig = create_topic_comparison_chart(
        topic_a_words,
        topic_b_words,
        topic_a,
        topic_b,
        height=500,
    )
    st.plotly_chart(fig, width='stretch')
    
    # Word tables
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Kata Topik {topic_a}")
        
        words_a_df = pd.DataFrame({
            "Peringkat": range(1, len(topic_a_words) + 1),
            "Kata": [w for w, _ in topic_a_words],
            "Bobot": [f"{w:.4f}" for _, w in topic_a_words],
        })
        st.dataframe(words_a_df, hide_index=True, width='stretch')
    
    with col2:
        st.markdown(f"### Kata Topik {topic_b}")
        
        words_b_df = pd.DataFrame({
            "Peringkat": range(1, len(topic_b_words) + 1),
            "Kata": [w for w, _ in topic_b_words],
            "Bobot": [f"{w:.4f}" for _, w in topic_b_words],
        })
        st.dataframe(words_b_df, hide_index=True, width='stretch')

with tab2:
    st.subheader("Word Cloud")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Topik {topic_a}")
        wc_a = create_wordcloud_figure(topic_a_words, colormap="Blues")
        st.image(f"data:image/png;base64,{wc_a}", width='stretch')
    
    with col2:
        st.markdown(f"### Topik {topic_b}")
        wc_b = create_wordcloud_figure(topic_b_words, colormap="Oranges")
        st.image(f"data:image/png;base64,{wc_b}", width='stretch')

with tab3:
    st.subheader("Analisis Tumpang Tindih")
    
    if analyzer:
        try:
            overlap = analyzer.get_topic_overlap(topic_a, topic_b, num_words=20)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Kemiripan Jaccard",
                    f"{overlap['jaccard_similarity']:.3f}",
                )
            
            with col2:
                st.metric(
                    "Kata Bersama",
                    len(overlap['shared_words']),
                )
            
            with col3:
                # Get distance
                distance_matrix = analyzer.compute_topic_distance_matrix()
                distance = distance_matrix[topic_a, topic_b]
                st.metric(
                    "Jarak JS",
                    f"{distance:.3f}",
                )
            
            st.markdown("---")
            
            # Shared words
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**🟢 Kata Bersama ({len(overlap['shared_words'])})**")
                if overlap['shared_words']:
                    for word in overlap['shared_words'][:10]:
                        st.write(f"• {word}")
                else:
                    st.info("Tidak ada kata bersama")
            
            with col2:
                st.markdown(f"**🔵 Unik di Topik {topic_a} ({len(overlap['unique_to_a'])})**")
                for word in overlap['unique_to_a'][:10]:
                    st.write(f"• {word}")
            
            with col3:
                st.markdown(f"**🟠 Unik di Topik {topic_b} ({len(overlap['unique_to_b'])})**")
                for word in overlap['unique_to_b'][:10]:
                    st.write(f"• {word}")
                    
        except Exception as e:
            st.error(f"Error menghitung tumpang tindih: {e}")

# Topic Distance Matrix
st.markdown("---")
st.subheader("📊 Matriks Jarak Topik")

try:
    if analyzer:
        distance_matrix = analyzer.compute_topic_distance_matrix()
        
        # Create heatmap
        labels = [f"T{i}" for i in range(num_topics)]
        fig = create_heatmap(
            distance_matrix,
            labels,
            labels,
            title="Jarak Topik (Jensen-Shannon)",
            colorscale="RdYlGn_r",
            height=500,
        )
        st.plotly_chart(fig, width='stretch')
        
        # Interpretation
        st.markdown("""
        **Cara membaca:**
        - **Nilai rendah** (hijau): Topik mirip
        - **Nilai tinggi** (merah): Topik berbeda
        - **Diagonal** selalu 0 (topik dibandingkan dengan dirinya sendiri)
        """)
        
except Exception as e:
    st.error(f"Error menghitung matriks jarak: {e}")

# Document comparison
st.markdown("---")
st.subheader("📚 Tumpang Tindih Dokumen")

if df is not None:
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            min_prob = st.slider(
                "Probabilitas minimum untuk dihitung sebagai dominan",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05,
                key="doc_overlap_prob",
            )
        
        # Count documents with high probability for each topic
        topic_a_col = f'topic_{topic_a}'
        topic_b_col = f'topic_{topic_b}'
        
        if topic_a_col in df.columns and topic_b_col in df.columns:
            docs_a = set(df[df[topic_a_col] >= min_prob].index)
            docs_b = set(df[df[topic_b_col] >= min_prob].index)
            
            docs_both = docs_a & docs_b
            docs_only_a = docs_a - docs_b
            docs_only_b = docs_b - docs_a
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"Dokumen Topik {topic_a}", len(docs_a))
            
            with col2:
                st.metric(f"Dokumen Topik {topic_b}", len(docs_b))
            
            with col3:
                st.metric("Dokumen Bersama", len(docs_both))
            
            # Venn diagram-like display
            st.markdown(f"""
            - **Hanya Topik {topic_a}**: {len(docs_only_a)} dokumen
            - **Hanya Topik {topic_b}**: {len(docs_only_b)} dokumen
            - **Kedua Topik**: {len(docs_both)} dokumen
            """)
            
    except Exception as e:
        st.warning(f"Tidak dapat menghitung tumpang tindih dokumen: {e}")
