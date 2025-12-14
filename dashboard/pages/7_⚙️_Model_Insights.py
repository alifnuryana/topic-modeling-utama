"""
Halaman Wawasan Model untuk Dashboard Pemodelan Topik.
"""

import streamlit as st
import sys
from pathlib import Path
import json

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dashboard.utils import load_model, load_data, load_processed_docs, check_model_loaded

st.set_page_config(
    page_title="Wawasan Model - Pemodelan Topik",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Wawasan Model")
st.markdown("Lihat konfigurasi model, metrik, dan diagnostik.")
st.markdown("---")

# Load model
if not check_model_loaded():
    st.stop()

model = load_model()
df = load_data()
processed_docs = load_processed_docs()

# Model Metadata
st.subheader("📊 Metadata Model")

col1, col2, col3, col4 = st.columns(4)

if model.metadata:
    meta = model.metadata
    
    with col1:
        st.metric("Jumlah Topik", meta.num_topics)
        st.metric("Dokumen", f"{meta.num_documents:,}")
    
    with col2:
        st.metric("Ukuran Kosakata", f"{meta.vocabulary_size:,}")
        st.metric("Skor Koherensi", f"{meta.coherence_score:.4f}")
    
    with col3:
        st.metric("Passes Pelatihan", meta.passes)
        st.metric("Iterasi", meta.iterations)
    
    with col4:
        st.metric("Alpha", meta.alpha)
        st.metric("Eta", meta.eta if meta.eta else "auto")
else:
    st.warning("Metadata model tidak tersedia")

# Training Configuration
st.markdown("---")
st.subheader("🔧 Konfigurasi Pelatihan")

try:
    from src.config import get_settings
    settings = get_settings()
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.markdown("**Parameter LDA:**")
        st.json({
            "num_topics": settings.lda_num_topics,
            "passes": settings.lda_passes,
            "iterations": settings.lda_iterations,
            "chunksize": settings.lda_chunksize,
            "alpha": settings.lda_alpha,
            "eta": settings.lda_eta,
            "random_state": settings.lda_random_state,
            "workers": settings.lda_workers,
        })
    
    with config_col2:
        st.markdown("**Filter Kamus:**")
        st.json({
            "no_below": settings.dict_no_below,
            "no_above": settings.dict_no_above,
            "keep_n": settings.dict_keep_n,
        })
        
        st.markdown("**Preprocessing:**")
        st.json({
            "min_word_length": settings.min_word_length,
            "max_word_length": settings.max_word_length,
            "min_doc_length": settings.min_doc_length,
            "use_stemming": settings.use_stemming,
            "use_bigrams": settings.use_bigrams,
            "use_trigrams": settings.use_trigrams,
        })
        
except Exception as e:
    st.warning(f"Tidak dapat memuat pengaturan: {e}")

# Vocabulary Analysis
st.markdown("---")
st.subheader("📝 Analisis Kosakata")

if model.dictionary:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Total Istilah", len(model.dictionary))
        
        # Get most frequent terms
        if hasattr(model.dictionary, 'cfs'):
            sorted_terms = sorted(
                model.dictionary.cfs.items(),
                key=lambda x: -x[1]
            )[:20]
            
            st.markdown("**20 Istilah Teratas berdasarkan Frekuensi:**")
            for term_id, freq in sorted_terms[:10]:
                term = model.dictionary[term_id]
                st.write(f"• {term}: {freq:,}")
    
    with col2:
        # Term frequency distribution
        if hasattr(model.dictionary, 'cfs') and model.dictionary.cfs:
            freqs = list(model.dictionary.cfs.values())
            
            st.markdown("**Statistik Frekuensi Istilah:**")
            stats_df = pd.DataFrame({
                "Statistik": ["Min", "Maks", "Rata-rata", "Median", "Std"],
                "Nilai": [
                    f"{min(freqs):,}",
                    f"{max(freqs):,}",
                    f"{np.mean(freqs):,.1f}",
                    f"{np.median(freqs):,.1f}",
                    f"{np.std(freqs):,.1f}",
                ]
            })
            st.dataframe(stats_df, hide_index=True, width='stretch')
else:
    st.info("Kamus tidak tersedia")

# Corpus Statistics
st.markdown("---")
st.subheader("📚 Statistik Korpus")

if processed_docs:
    col1, col2 = st.columns(2)
    
    with col1:
        total_tokens = sum(len(doc) for doc in processed_docs)
        unique_tokens = len(set(token for doc in processed_docs for token in doc))
        
        st.metric("Total Dokumen", len(processed_docs))
        st.metric("Total Token", f"{total_tokens:,}")
        st.metric("Token Unik", f"{unique_tokens:,}")
    
    with col2:
        doc_lengths = [len(doc) for doc in processed_docs]
        
        st.metric("Rata-rata Token/Dokumen", f"{np.mean(doc_lengths):.1f}")
        st.metric("Panjang Dokumen Minimum", min(doc_lengths))
        st.metric("Panjang Dokumen Maksimum", max(doc_lengths))
else:
    if df is not None:
        st.metric("Total Dokumen", len(df))
    else:
        st.info("Korpus tidak tersedia")

# Topic Details
st.markdown("---")
st.subheader("📋 Detail Topik")

topics = model.get_topics(num_words=15)

topic_data = []
for topic in topics:
    topic_data.append({
        "ID Topik": topic.topic_id,
        "Label": topic.label[:50] + "..." if len(topic.label) > 50 else topic.label,
        "Kata Teratas": ", ".join(topic.top_words[:8]),
    })

topic_df = pd.DataFrame(topic_data)
st.dataframe(topic_df, hide_index=True, width='stretch')

# Per-topic metrics
if model.metadata and df is not None:
    st.markdown("### Metrik Topik")
    
    metrics_data = []
    for t in range(model.model.num_topics):
        col_name = f'topic_{t}'
        if col_name in df.columns:
            probs = df[col_name]
            dominant = (df.get('dominant_topic', pd.Series()) == t).sum() if 'dominant_topic' in df.columns else 0
            
            metrics_data.append({
                "Topik": t,
                "Prob Rata-rata": f"{probs.mean():.4f}",
                "Prob Std": f"{probs.std():.4f}",
                "Prob Maks": f"{probs.max():.4f}",
                "Dok Dominan": dominant,
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, hide_index=True, width='stretch')

# File Paths
st.markdown("---")
st.subheader("📁 Lokasi File")

try:
    from src.config import get_settings
    settings = get_settings()
    
    files = {
        "Direktori model": str(settings.models_dir),
        "Data terproses": str(settings.processed_data_dir),
        "Output": str(settings.outputs_dir),
    }
    
    for name, path in files.items():
        st.markdown(f"**{name}**: `{path}`")
        
except Exception as e:
    st.warning(f"Tidak dapat memuat path: {e}")

# Export Options
st.markdown("---")
st.subheader("📥 Opsi Ekspor")

col1, col2, col3 = st.columns(3)

with col1:
    if model.metadata:
        metadata_json = json.dumps(model.metadata.to_dict(), indent=2)
        st.download_button(
            label="📥 Unduh Metadata (JSON)",
            data=metadata_json,
            file_name="metadata_model.json",
            mime="application/json",
        )

with col2:
    topics_data = [t.to_dict() for t in topics]
    topics_json = json.dumps(topics_data, indent=2)
    st.download_button(
        label="📥 Unduh Topik (JSON)",
        data=topics_json,
        file_name="topik.json",
        mime="application/json",
    )

with col3:
    if df is not None:
        st.download_button(
            label="📥 Unduh Data Lengkap (CSV)",
            data=df.to_csv(index=False),
            file_name="matriks_topik_dokumen.csv",
            mime="text/csv",
        )
