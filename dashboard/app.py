"""
Topic Modeling Dashboard - Main Application

Streamlit multi-page dashboard for exploring LDA topic modeling results.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Topic Modeling Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
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
    st.markdown('<p class="main-header">📊 Topic Modeling Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Explore topics discovered from academic paper metadata using LDA</p>',
        unsafe_allow_html=True
    )
    
    # Welcome section
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Welcome!")
        st.markdown("""
        This dashboard provides an interactive interface to explore the results of 
        **Latent Dirichlet Allocation (LDA)** topic modeling on academic paper metadata.
        
        **Available Features:**
        - 📊 **Topic Explorer** - Browse and visualize discovered topics
        - 📄 **Document Browser** - Search and filter documents by topic
        - 🔍 **Similarity Search** - Find similar documents or analyze new text
        - 📈 **Trend Analysis** - See how topics evolve over time
        - 🎯 **Topic Comparison** - Compare topics side by side
        - ⚙️ **Model Insights** - View model metrics and configuration
        
        Use the sidebar navigation to explore different features.
        """)
    
    with col2:
        st.markdown("### 📚 Quick Stats")
        
        # Try to load quick stats
        try:
            from dashboard.utils import load_model, load_data
            
            model = load_model()
            df = load_data()
            
            if model and df is not None:
                st.metric("Topics", model.model.num_topics)
                st.metric("Documents", f"{len(df):,}")
                if model.metadata:
                    st.metric("Coherence", f"{model.metadata.coherence_score:.4f}")
            else:
                st.info("Model not loaded yet. Run the notebooks first.")
        except Exception as e:
            st.info("Run the Jupyter notebooks to train a model first.")
    
    st.markdown("---")
    
    # Getting started
    st.markdown("### 🚀 Getting Started")
    
    st.markdown("""
    If you haven't trained a model yet, follow these steps:
    
    1. **Install dependencies**: `uv sync`
    2. **Run notebooks** in order:
       - `01_data_collection.ipynb` - Harvest data
       - `01b_eda_raw_data.ipynb` - Explore raw data
       - `02_data_cleaning.ipynb` - Clean the data
       - `02b_eda_clean_data.ipynb` - Explore cleaned data
       - `03_preprocessing.ipynb` - Preprocess text
       - `04_lda_modeling.ipynb` - Train LDA model
       - `05_analysis_visualization.ipynb` - Generate visualizations
    3. **Return to this dashboard** to explore results interactively!
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>Topic Modeling Dashboard • Built with Streamlit</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
