"""
Home page for the Topic Modeling Dashboard.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dashboard.utils import load_model, load_data, check_model_loaded

st.set_page_config(
    page_title="Home - Topic Modeling",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Home")
st.markdown("---")

# Check if model is loaded
model = load_model()
df = load_data()

if model is None:
    st.warning("⚠️ Model not loaded. Please run the notebooks first to train a model.")
    
    st.markdown("""
    ### Getting Started
    
    Follow these steps to train a topic model:
    
    1. **Install dependencies**: Run `uv sync` in the terminal
    2. **Run notebooks** in order:
       - `01_data_collection.ipynb` - Harvest data from repository
       - `01b_eda_raw_data.ipynb` - Explore raw data
       - `02_data_cleaning.ipynb` - Clean the data
       - `02b_eda_clean_data.ipynb` - Explore cleaned data
       - `03_preprocessing.ipynb` - Preprocess text
       - `04_lda_modeling.ipynb` - Train LDA model
       - `05_analysis_visualization.ipynb` - Generate visualizations
    3. **Return here** to explore the results!
    """)
else:
    # Display model info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Topics", model.model.num_topics)
    
    with col2:
        if df is not None:
            st.metric("📚 Documents", f"{len(df):,}")
    
    with col3:
        if model.metadata:
            st.metric("📈 Coherence", f"{model.metadata.coherence_score:.4f}")
    
    with col4:
        if model.metadata:
            st.metric("📝 Vocabulary", f"{model.metadata.vocabulary_size:,}")
    
    st.markdown("---")
    
    # Topics overview
    st.subheader("📋 Topics Overview")
    
    topics = model.get_topics(num_words=8)
    
    # Display topics in a grid
    cols = st.columns(2)
    
    for i, topic in enumerate(topics):
        with cols[i % 2]:
            words = ", ".join(topic.top_words[:6])
            st.markdown(f"**Topic {topic.topic_id}**: {words}")
    
    st.markdown("---")
    
    # Quick navigation
    st.subheader("🚀 Quick Navigation")
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        st.markdown("""
        **📊 Topic Explorer**
        
        Browse topics, view word clouds, and explore top words.
        """)
    
    with nav_col2:
        st.markdown("""
        **📄 Document Browser**
        
        Search and filter documents by topic or keyword.
        """)
    
    with nav_col3:
        st.markdown("""
        **🔍 Similarity Search**
        
        Find similar documents or analyze new text.
        """)
    
    # Dataset info
    if df is not None:
        st.markdown("---")
        st.subheader("📊 Dataset Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'year' in df.columns:
                year_min = int(df['year'].min()) if df['year'].notna().any() else "N/A"
                year_max = int(df['year'].max()) if df['year'].notna().any() else "N/A"
                st.write(f"**Date Range**: {year_min} - {year_max}")
            
            if 'title' in df.columns:
                avg_title_len = df['title'].str.len().mean()
                st.write(f"**Avg Title Length**: {avg_title_len:.0f} characters")
        
        with col2:
            if 'dominant_topic' in df.columns:
                most_common_topic = df['dominant_topic'].mode().iloc[0]
                st.write(f"**Most Common Topic**: Topic {int(most_common_topic)}")
            
            if 'abstract' in df.columns:
                avg_abstract_len = df['abstract'].str.split().str.len().mean()
                st.write(f"**Avg Abstract Length**: {avg_abstract_len:.0f} words")
