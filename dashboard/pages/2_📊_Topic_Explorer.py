"""
Topic Explorer page for the Topic Modeling Dashboard.
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
    get_analyzer, get_visualizer, check_model_loaded
)
from dashboard.components.charts import (
    create_wordcloud_figure, create_topic_bar_chart,
    create_topic_prevalence_chart
)
from dashboard.components.filters import topic_selector

st.set_page_config(
    page_title="Topic Explorer - Topic Modeling",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Topic Explorer")
st.markdown("Explore the discovered topics, view word clouds, and analyze topic content.")
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
    st.header("Options")
    
    selected_topic = topic_selector(
        num_topics,
        key="explorer_topic",
        label="Select Topic to Explore",
        include_all=False,
    )
    
    num_words = st.slider(
        "Number of Words",
        min_value=5,
        max_value=30,
        value=15,
        key="num_words",
    )

# Main content
tab1, tab2, tab3 = st.tabs(["📈 Topic Overview", "☁️ Word Cloud", "🔤 Word Details"])

with tab1:
    st.subheader("Topic Prevalence")
    
    if analyzer:
        try:
            prevalence = analyzer.compute_topic_prevalence()
            fig = create_topic_prevalence_chart(prevalence)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            st.markdown("### Topic Summary")
            
            cols = st.columns(4)
            for i, row in prevalence.iterrows():
                col_idx = i % 4
                with cols[col_idx]:
                    topic_id = int(row['topic_id'])
                    mean_prob = row['mean_probability']
                    num_dominant = int(row['num_dominant'])
                    
                    st.metric(
                        f"Topic {topic_id}",
                        f"{mean_prob:.3f}",
                        f"{num_dominant} docs",
                    )
        except Exception as e:
            st.error(f"Error computing prevalence: {e}")
    
    # All topics list
    st.markdown("### All Topics")
    
    topics = model.get_topics(num_words=8)
    for topic in topics:
        with st.expander(f"Topic {topic.topic_id}: {', '.join(topic.top_words[:5])}"):
            st.write(f"**Top words**: {', '.join(topic.top_words)}")

with tab2:
    st.subheader(f"Word Cloud - Topic {selected_topic}")
    
    # Get word weights
    words = model.get_topic_words(selected_topic, num_words=50)
    
    if words:
        # Create word cloud
        wc_base64 = create_wordcloud_figure(words)
        st.image(f"data:image/png;base64,{wc_base64}", use_container_width=True)
    else:
        st.info("No words found for this topic")
    
    # Color options
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Topic {selected_topic} Summary**")
        topic_info = topics[selected_topic]
        st.write(f"Top words: {', '.join(topic_info.top_words[:10])}")
    
    with col2:
        if analyzer and df is not None:
            try:
                prevalence = analyzer.compute_topic_prevalence()
                topic_prev = prevalence[prevalence['topic_id'] == selected_topic].iloc[0]
                st.metric("Mean Probability", f"{topic_prev['mean_probability']:.4f}")
                st.metric("Documents (Dominant)", int(topic_prev['num_dominant']))
            except:
                pass

with tab3:
    st.subheader(f"Word Details - Topic {selected_topic}")
    
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
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.markdown("### Word Weights Table")
        
        import pandas as pd
        word_df = pd.DataFrame({
            "Rank": range(1, len(words) + 1),
            "Word": word_list,
            "Weight": weight_list,
        })
        st.dataframe(word_df, use_container_width=True, hide_index=True)
    else:
        st.info("No words found for this topic")

# pyLDAvis section
st.markdown("---")
st.subheader("📊 Interactive Topic Visualization (pyLDAvis)")

try:
    from src.config import get_settings
    settings = get_settings()
    pyldavis_path = settings.outputs_dir / "pyldavis.html"
    
    if pyldavis_path.exists():
        with open(pyldavis_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.info("pyLDAvis visualization not found. Run the 05_analysis_visualization notebook to generate it.")
except Exception as e:
    st.warning(f"Could not load pyLDAvis: {e}")
