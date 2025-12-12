"""
Topic Comparison page for the Topic Modeling Dashboard.
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

from dashboard.utils import load_model, load_data, get_analyzer, get_visualizer, check_model_loaded
from dashboard.components.charts import (
    create_topic_comparison_chart, create_wordcloud_figure, create_heatmap
)
from dashboard.components.filters import topic_selector

st.set_page_config(
    page_title="Topic Comparison - Topic Modeling",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Topic Comparison")
st.markdown("Compare topics side by side to understand their similarities and differences.")
st.markdown("---")

# Load model
if not check_model_loaded():
    st.stop()

model = load_model()
df = load_data()
analyzer = get_analyzer(model)
visualizer = get_visualizer(model)

num_topics = model.model.num_topics

# Sidebar
with st.sidebar:
    st.header("Select Topics to Compare")
    
    topic_a = st.selectbox(
        "Topic A",
        options=list(range(num_topics)),
        index=0,
        format_func=lambda x: f"Topic {x}",
        key="compare_topic_a",
    )
    
    topic_b = st.selectbox(
        "Topic B",
        options=list(range(num_topics)),
        index=min(1, num_topics - 1),
        format_func=lambda x: f"Topic {x}",
        key="compare_topic_b",
    )
    
    st.markdown("---")
    
    num_words = st.slider(
        "Number of Words",
        min_value=5,
        max_value=20,
        value=10,
        key="compare_num_words",
    )

# Get topic data
topic_a_words = model.get_topic_words(topic_a, num_words)
topic_b_words = model.get_topic_words(topic_b, num_words)

# Main comparison
tab1, tab2, tab3 = st.tabs(["📊 Word Comparison", "☁️ Word Clouds", "🔗 Overlap Analysis"])

with tab1:
    st.subheader("Side-by-Side Comparison")
    
    # Comparison chart
    fig = create_topic_comparison_chart(
        topic_a_words,
        topic_b_words,
        topic_a,
        topic_b,
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Word tables
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Topic {topic_a} Words")
        
        words_a_df = pd.DataFrame({
            "Rank": range(1, len(topic_a_words) + 1),
            "Word": [w for w, _ in topic_a_words],
            "Weight": [f"{w:.4f}" for _, w in topic_a_words],
        })
        st.dataframe(words_a_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown(f"### Topic {topic_b} Words")
        
        words_b_df = pd.DataFrame({
            "Rank": range(1, len(topic_b_words) + 1),
            "Word": [w for w, _ in topic_b_words],
            "Weight": [f"{w:.4f}" for _, w in topic_b_words],
        })
        st.dataframe(words_b_df, hide_index=True, use_container_width=True)

with tab2:
    st.subheader("Word Clouds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Topic {topic_a}")
        wc_a = create_wordcloud_figure(topic_a_words, colormap="Blues")
        st.image(f"data:image/png;base64,{wc_a}", use_container_width=True)
    
    with col2:
        st.markdown(f"### Topic {topic_b}")
        wc_b = create_wordcloud_figure(topic_b_words, colormap="Oranges")
        st.image(f"data:image/png;base64,{wc_b}", use_container_width=True)

with tab3:
    st.subheader("Overlap Analysis")
    
    if analyzer:
        try:
            overlap = analyzer.get_topic_overlap(topic_a, topic_b, num_words=20)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Jaccard Similarity",
                    f"{overlap['jaccard_similarity']:.3f}",
                )
            
            with col2:
                st.metric(
                    "Shared Words",
                    len(overlap['shared_words']),
                )
            
            with col3:
                # Get distance
                distance_matrix = analyzer.compute_topic_distance_matrix()
                distance = distance_matrix[topic_a, topic_b]
                st.metric(
                    "JS Distance",
                    f"{distance:.3f}",
                )
            
            st.markdown("---")
            
            # Shared words
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**🟢 Shared Words ({len(overlap['shared_words'])})**")
                if overlap['shared_words']:
                    for word in overlap['shared_words'][:10]:
                        st.write(f"• {word}")
                else:
                    st.info("No shared words")
            
            with col2:
                st.markdown(f"**🔵 Unique to Topic {topic_a} ({len(overlap['unique_to_a'])})**")
                for word in overlap['unique_to_a'][:10]:
                    st.write(f"• {word}")
            
            with col3:
                st.markdown(f"**🟠 Unique to Topic {topic_b} ({len(overlap['unique_to_b'])})**")
                for word in overlap['unique_to_b'][:10]:
                    st.write(f"• {word}")
                    
        except Exception as e:
            st.error(f"Error computing overlap: {e}")

# Topic Distance Matrix
st.markdown("---")
st.subheader("📊 Topic Distance Matrix")

try:
    if analyzer:
        distance_matrix = analyzer.compute_topic_distance_matrix()
        
        # Create heatmap
        labels = [f"T{i}" for i in range(num_topics)]
        fig = create_heatmap(
            distance_matrix,
            labels,
            labels,
            title="Topic Distance (Jensen-Shannon)",
            colorscale="RdYlGn_r",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("""
        **How to interpret:**
        - **Lower values** (green): Topics are similar
        - **Higher values** (red): Topics are different
        - **Diagonal** is always 0 (topic compared to itself)
        """)
        
except Exception as e:
    st.error(f"Error computing distance matrix: {e}")

# Document comparison
st.markdown("---")
st.subheader("📚 Document Overlap")

if df is not None:
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            min_prob = st.slider(
                "Min probability to count as dominant",
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
                st.metric(f"Topic {topic_a} Docs", len(docs_a))
            
            with col2:
                st.metric(f"Topic {topic_b} Docs", len(docs_b))
            
            with col3:
                st.metric("Shared Docs", len(docs_both))
            
            # Venn diagram-like display
            st.markdown(f"""
            - **Only Topic {topic_a}**: {len(docs_only_a)} documents
            - **Only Topic {topic_b}**: {len(docs_only_b)} documents
            - **Both Topics**: {len(docs_both)} documents
            """)
            
    except Exception as e:
        st.warning(f"Could not compute document overlap: {e}")
