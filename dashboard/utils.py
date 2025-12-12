"""
Dashboard utility functions.

Provides caching and loading utilities for the Streamlit dashboard.
"""

import pickle
from pathlib import Path
from typing import Optional
import sys

import pandas as pd
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.lda_model import LDATopicModel
from src.preprocessor import IndonesianPreprocessor
from src.analysis import TopicAnalyzer
from src.visualizations import TopicVisualizer


@st.cache_resource
def load_model() -> Optional[LDATopicModel]:
    """Load the trained LDA model with caching."""
    try:
        settings = get_settings()
        model = LDATopicModel(settings)
        model.load()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """Load the topic-document matrix with caching."""
    try:
        settings = get_settings()
        data_path = settings.processed_data_dir / settings.topic_document_matrix_file
        
        if data_path.exists():
            return pd.read_csv(data_path)
        
        # Try loading from corpus
        corpus_path = settings.processed_data_dir / settings.processed_corpus_file
        if corpus_path.exists():
            with open(corpus_path, 'rb') as f:
                corpus_data = pickle.load(f)
            return corpus_data.get('dataframe')
        
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def load_processed_docs() -> Optional[list]:
    """Load processed documents with caching."""
    try:
        settings = get_settings()
        corpus_path = settings.processed_data_dir / settings.processed_corpus_file
        
        if corpus_path.exists():
            with open(corpus_path, 'rb') as f:
                corpus_data = pickle.load(f)
            return corpus_data.get('documents')
        
        return None
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return None


@st.cache_resource
def load_preprocessor() -> Optional[IndonesianPreprocessor]:
    """Load the preprocessor with caching."""
    try:
        settings = get_settings()
        preprocessor_path = settings.processed_data_dir / 'preprocessor.pkl'
        
        if preprocessor_path.exists():
            preprocessor = IndonesianPreprocessor(settings)
            preprocessor.load(preprocessor_path)
            return preprocessor
        
        # Return a new preprocessor if saved one doesn't exist
        return IndonesianPreprocessor(settings)
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        return None


@st.cache_resource
def get_analyzer(_model: LDATopicModel) -> Optional[TopicAnalyzer]:
    """Get topic analyzer instance."""
    try:
        settings = get_settings()
        analyzer = TopicAnalyzer(_model, settings)
        
        # Set document data
        df = load_data()
        if df is not None and 'tokens' not in df.columns:
            # Load tokens from corpus
            docs = load_processed_docs()
            if docs and len(docs) == len(df):
                df['tokens'] = docs
        
        if df is not None:
            analyzer.set_document_data(df, tokens_column='tokens')
        
        return analyzer
    except Exception as e:
        st.error(f"Error creating analyzer: {e}")
        return None


@st.cache_resource
def get_visualizer(_model: LDATopicModel) -> TopicVisualizer:
    """Get topic visualizer instance."""
    settings = get_settings()
    return TopicVisualizer(_model, settings)


def get_topic_colors(num_topics: int) -> list:
    """Get a list of colors for topics."""
    import plotly.colors as pc
    
    if num_topics <= 10:
        colors = pc.qualitative.Plotly[:num_topics]
    else:
        colors = pc.sample_colorscale(
            'Viridis', 
            [i / (num_topics - 1) for i in range(num_topics)]
        )
    
    return colors


def format_topic_label(topic_id: int, words: list, max_words: int = 5) -> str:
    """Format a topic label from its top words."""
    word_str = ", ".join(words[:max_words])
    return f"Topic {topic_id}: {word_str}"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def check_model_loaded() -> bool:
    """Check if model is loaded and show message if not."""
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Model not loaded. Please run the notebooks first to train a model.")
        st.info("""
        **To train a model:**
        1. Run `01_data_collection.ipynb` to harvest data
        2. Run `02_data_cleaning.ipynb` to clean the data
        3. Run `03_preprocessing.ipynb` to preprocess text
        4. Run `04_lda_modeling.ipynb` to train the LDA model
        """)
        return False
    
    return True
