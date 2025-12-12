"""
Model Insights page for the Topic Modeling Dashboard.
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
    page_title="Model Insights - Topic Modeling",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Model Insights")
st.markdown("View model configuration, metrics, and diagnostics.")
st.markdown("---")

# Load model
if not check_model_loaded():
    st.stop()

model = load_model()
df = load_data()
processed_docs = load_processed_docs()

# Model Metadata
st.subheader("📊 Model Metadata")

col1, col2, col3, col4 = st.columns(4)

if model.metadata:
    meta = model.metadata
    
    with col1:
        st.metric("Number of Topics", meta.num_topics)
        st.metric("Documents", f"{meta.num_documents:,}")
    
    with col2:
        st.metric("Vocabulary Size", f"{meta.vocabulary_size:,}")
        st.metric("Coherence Score", f"{meta.coherence_score:.4f}")
    
    with col3:
        st.metric("Training Passes", meta.passes)
        st.metric("Iterations", meta.iterations)
    
    with col4:
        st.metric("Alpha", meta.alpha)
        st.metric("Eta", meta.eta if meta.eta else "auto")
else:
    st.warning("Model metadata not available")

# Training Configuration
st.markdown("---")
st.subheader("🔧 Training Configuration")

try:
    from src.config import get_settings
    settings = get_settings()
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.markdown("**LDA Parameters:**")
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
        st.markdown("**Dictionary Filters:**")
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
    st.warning(f"Could not load settings: {e}")

# Vocabulary Analysis
st.markdown("---")
st.subheader("📝 Vocabulary Analysis")

if model.dictionary:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Total Terms", len(model.dictionary))
        
        # Get most frequent terms
        if hasattr(model.dictionary, 'cfs'):
            sorted_terms = sorted(
                model.dictionary.cfs.items(),
                key=lambda x: -x[1]
            )[:20]
            
            st.markdown("**Top 20 Terms by Frequency:**")
            for term_id, freq in sorted_terms[:10]:
                term = model.dictionary[term_id]
                st.write(f"• {term}: {freq:,}")
    
    with col2:
        # Term frequency distribution
        if hasattr(model.dictionary, 'cfs') and model.dictionary.cfs:
            freqs = list(model.dictionary.cfs.values())
            
            st.markdown("**Term Frequency Statistics:**")
            stats_df = pd.DataFrame({
                "Statistic": ["Min", "Max", "Mean", "Median", "Std"],
                "Value": [
                    f"{min(freqs):,}",
                    f"{max(freqs):,}",
                    f"{np.mean(freqs):,.1f}",
                    f"{np.median(freqs):,.1f}",
                    f"{np.std(freqs):,.1f}",
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
else:
    st.info("Dictionary not available")

# Corpus Statistics
st.markdown("---")
st.subheader("📚 Corpus Statistics")

if processed_docs:
    col1, col2 = st.columns(2)
    
    with col1:
        total_tokens = sum(len(doc) for doc in processed_docs)
        unique_tokens = len(set(token for doc in processed_docs for token in doc))
        
        st.metric("Total Documents", len(processed_docs))
        st.metric("Total Tokens", f"{total_tokens:,}")
        st.metric("Unique Tokens", f"{unique_tokens:,}")
    
    with col2:
        doc_lengths = [len(doc) for doc in processed_docs]
        
        st.metric("Avg Tokens/Document", f"{np.mean(doc_lengths):.1f}")
        st.metric("Min Document Length", min(doc_lengths))
        st.metric("Max Document Length", max(doc_lengths))
else:
    if df is not None:
        st.metric("Total Documents", len(df))
    else:
        st.info("Corpus not available")

# Topic Details
st.markdown("---")
st.subheader("📋 Topic Details")

topics = model.get_topics(num_words=15)

topic_data = []
for topic in topics:
    topic_data.append({
        "Topic ID": topic.topic_id,
        "Label": topic.label[:50] + "..." if len(topic.label) > 50 else topic.label,
        "Top Words": ", ".join(topic.top_words[:8]),
    })

topic_df = pd.DataFrame(topic_data)
st.dataframe(topic_df, hide_index=True, use_container_width=True)

# Per-topic metrics
if model.metadata and df is not None:
    st.markdown("### Topic Metrics")
    
    metrics_data = []
    for t in range(model.model.num_topics):
        col_name = f'topic_{t}'
        if col_name in df.columns:
            probs = df[col_name]
            dominant = (df.get('dominant_topic', pd.Series()) == t).sum() if 'dominant_topic' in df.columns else 0
            
            metrics_data.append({
                "Topic": t,
                "Mean Prob": f"{probs.mean():.4f}",
                "Std Prob": f"{probs.std():.4f}",
                "Max Prob": f"{probs.max():.4f}",
                "Dominant Docs": dominant,
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

# File Paths
st.markdown("---")
st.subheader("📁 File Locations")

try:
    from src.config import get_settings
    settings = get_settings()
    
    files = {
        "Model directory": str(settings.models_dir),
        "Processed data": str(settings.processed_data_dir),
        "Outputs": str(settings.outputs_dir),
    }
    
    for name, path in files.items():
        st.markdown(f"**{name}**: `{path}`")
        
except Exception as e:
    st.warning(f"Could not load paths: {e}")

# Export Options
st.markdown("---")
st.subheader("📥 Export Options")

col1, col2, col3 = st.columns(3)

with col1:
    if model.metadata:
        metadata_json = json.dumps(model.metadata.to_dict(), indent=2)
        st.download_button(
            label="📥 Download Metadata (JSON)",
            data=metadata_json,
            file_name="model_metadata.json",
            mime="application/json",
        )

with col2:
    topics_data = [t.to_dict() for t in topics]
    topics_json = json.dumps(topics_data, indent=2)
    st.download_button(
        label="📥 Download Topics (JSON)",
        data=topics_json,
        file_name="topics.json",
        mime="application/json",
    )

with col3:
    if df is not None:
        st.download_button(
            label="📥 Download Full Data (CSV)",
            data=df.to_csv(index=False),
            file_name="topic_document_matrix.csv",
            mime="text/csv",
        )
