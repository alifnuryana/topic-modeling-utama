"""
Halaman Pencarian Kemiripan untuk Dashboard Pemodelan Topik.
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

from dashboard.utils import (
    load_model, load_data, load_preprocessor, load_processed_docs,
    get_analyzer, check_model_loaded
)
from dashboard.components.charts import create_similarity_chart
from dashboard.components.filters import num_results_selector

st.set_page_config(
    page_title="Pencarian Kemiripan - Pemodelan Topik",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Pencarian Kemiripan")
st.markdown("Temukan dokumen serupa atau analisis distribusi topik dari teks baru.")
st.markdown("---")

# Load model
if not check_model_loaded():
    st.stop()

model = load_model()
df = load_data()
analyzer = get_analyzer(model)
preprocessor = load_preprocessor()
processed_docs = load_processed_docs()

if df is None:
    st.error("Tidak dapat memuat data dokumen.")
    st.stop()

num_topics = model.model.num_topics

# Tabs for different search modes
tab1, tab2 = st.tabs(["📄 Temukan Dokumen Serupa", "📝 Analisis Teks Baru"])

with tab1:
    st.subheader("Temukan Dokumen Serupa dengan Dokumen yang Dipilih")
    
    # Document selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create options from titles
        doc_options = {
            f"{i}: {title[:80]}..." if len(str(title)) > 80 else f"{i}: {title}": i
            for i, title in enumerate(df['title'].values[:500])  # Limit to first 500 for performance
        }
        
        selected_doc_str = st.selectbox(
            "Pilih Dokumen",
            options=list(doc_options.keys()),
            key="sim_doc_select",
        )
        
        selected_doc_idx = doc_options[selected_doc_str]
    
    with col2:
        num_similar = num_results_selector(
            key="num_similar",
            label="Jumlah Hasil",
            options=[5, 10, 15, 20],
            default_index=1,
        )
    
    # Display selected document
    selected_doc = df.iloc[selected_doc_idx]
    
    with st.expander("📄 Dokumen Terpilih", expanded=True):
        st.markdown(f"**Judul**: {selected_doc['title']}")
        if 'authors' in selected_doc:
            st.markdown(f"**Penulis**: {selected_doc['authors']}")
        if 'year' in selected_doc and pd.notna(selected_doc['year']):
            st.markdown(f"**Tahun**: {int(selected_doc['year'])}")
        st.markdown("---")
        st.markdown("**Abstrak**:")
        st.write(selected_doc['abstract'][:500] + "..." if len(str(selected_doc['abstract'])) > 500 else selected_doc['abstract'])
    
    # Find similar documents
    if st.button("🔍 Temukan Dokumen Serupa", key="find_similar"):
        with st.spinner("Mencari dokumen serupa..."):
            try:
                similar_docs = analyzer.find_similar_documents(selected_doc_idx, top_n=num_similar)
                
                if len(similar_docs) > 0:
                    st.success(f"Ditemukan {len(similar_docs)} dokumen serupa")
                    
                    # Display results
                    for rank, (_, row) in enumerate(similar_docs.iterrows(), 1):
                        similarity = row['similarity']
                        
                        # Color based on similarity
                        if similarity > 0.7:
                            icon = "🟢"
                        elif similarity > 0.4:
                            icon = "🟡"
                        else:
                            icon = "🔴"
                        
                        with st.expander(f"{icon} #{rank} - Kemiripan: {similarity:.3f} - {row['title'][:70]}..."):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Judul**: {row['title']}")
                                if 'authors' in row:
                                    st.markdown(f"**Penulis**: {row['authors']}")
                                if 'year' in row and pd.notna(row['year']):
                                    st.markdown(f"**Tahun**: {int(row['year'])}")
                                st.markdown("---")
                                abstract = str(row['abstract'])[:400]
                                st.write(abstract + "..." if len(str(row['abstract'])) > 400 else row['abstract'])
                            
                            with col2:
                                st.metric("Kemiripan", f"{similarity:.3f}")
                                if 'dominant_topic' in row:
                                    st.metric("Topik Dominan", f"Topik {int(row['dominant_topic'])}")
                else:
                    st.info("Tidak ditemukan dokumen serupa")
            except Exception as e:
                st.error(f"Error mencari dokumen serupa: {e}")

with tab2:
    st.subheader("Analisis Distribusi Topik dari Teks Baru")
    st.markdown("Masukkan teks di bawah ini untuk melihat distribusi topiknya berdasarkan model yang telah dilatih.")
    
    # Text input
    new_text = st.text_area(
        "Masukkan teks untuk dianalisis",
        height=200,
        placeholder="Tempel atau ketik teks Anda di sini...",
        key="new_text_input",
    )
    
    if st.button("🔍 Analisis Teks", key="analyze_text"):
        if not new_text.strip():
            st.warning("Silakan masukkan teks untuk dianalisis.")
        else:
            with st.spinner("Menganalisis teks..."):
                try:
                    # Infer topics
                    result = model.infer_topics(
                        new_text,
                        preprocessor=preprocessor,
                        num_words=10,
                    )
                    
                    st.success("Analisis selesai!")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### Info Preprocessing")
                        st.metric("Token", result['num_tokens'])
                        
                        with st.expander("Lihat token"):
                            st.write(result['tokens'][:50])
                            if len(result['tokens']) > 50:
                                st.caption(f"... dan {len(result['tokens']) - 50} lainnya")
                    
                    with col2:
                        st.markdown("### Topik Teratas")
                        
                        for topic_info in result['topics'][:5]:
                            topic_id = topic_info['topic_id']
                            prob = topic_info['probability']
                            words = ", ".join(topic_info['top_words'][:5])
                            
                            st.progress(
                                prob,
                                text=f"Topik {topic_id}: {prob:.1%}",
                            )
                            st.caption(f"Kata: {words}")
                    
                    # Find similar based on this text
                    st.markdown("---")
                    st.markdown("### Dokumen Serupa")
                    
                    try:
                        similar_docs = analyzer.find_similar_by_text(
                            new_text,
                            preprocessor=preprocessor,
                            top_n=5,
                        )
                        
                        if len(similar_docs) > 0:
                            for rank, (_, row) in enumerate(similar_docs.iterrows(), 1):
                                similarity = row['similarity']
                                st.markdown(f"**{rank}. (Kemiripan: {similarity:.3f})** {row['title'][:80]}...")
                        else:
                            st.info("Tidak ditemukan dokumen serupa")
                    except Exception as e:
                        st.warning(f"Tidak dapat menemukan dokumen serupa: {e}")
                        
                except Exception as e:
                    st.error(f"Error menganalisis teks: {e}")

# Sidebar with tips
with st.sidebar:
    st.header("💡 Tips")
    
    st.markdown("""
    **Menemukan Dokumen Serupa:**
    - Pilih dokumen dari dropdown
    - Pencarian menggunakan kemiripan distribusi topik
    - Skor kemiripan lebih tinggi berarti topik lebih mirip
    
    **Menganalisis Teks Baru:**
    - Masukkan atau tempel teks apapun
    - Bekerja paling baik dengan abstrak akademik
    - Teks lebih panjang memberikan hasil lebih baik
    - Menggunakan preprocessing yang sama dengan pelatihan
    """)
