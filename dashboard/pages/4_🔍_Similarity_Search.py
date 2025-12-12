"""
Similarity Search page for the Topic Modeling Dashboard.
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
    page_title="Similarity Search - Topic Modeling",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Similarity Search")
st.markdown("Find similar documents or analyze the topic distribution of new text.")
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
    st.error("Could not load document data.")
    st.stop()

num_topics = model.model.num_topics

# Tabs for different search modes
tab1, tab2 = st.tabs(["📄 Find Similar Documents", "📝 Analyze New Text"])

with tab1:
    st.subheader("Find Documents Similar to a Selected Document")
    
    # Document selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create options from titles
        doc_options = {
            f"{i}: {title[:80]}..." if len(str(title)) > 80 else f"{i}: {title}": i
            for i, title in enumerate(df['title'].values[:500])  # Limit to first 500 for performance
        }
        
        selected_doc_str = st.selectbox(
            "Select a Document",
            options=list(doc_options.keys()),
            key="sim_doc_select",
        )
        
        selected_doc_idx = doc_options[selected_doc_str]
    
    with col2:
        num_similar = num_results_selector(
            key="num_similar",
            label="Number of Results",
            options=[5, 10, 15, 20],
            default_index=1,
        )
    
    # Display selected document
    selected_doc = df.iloc[selected_doc_idx]
    
    with st.expander("📄 Selected Document", expanded=True):
        st.markdown(f"**Title**: {selected_doc['title']}")
        if 'authors' in selected_doc:
            st.markdown(f"**Authors**: {selected_doc['authors']}")
        if 'year' in selected_doc and pd.notna(selected_doc['year']):
            st.markdown(f"**Year**: {int(selected_doc['year'])}")
        st.markdown("---")
        st.markdown("**Abstract**:")
        st.write(selected_doc['abstract'][:500] + "..." if len(str(selected_doc['abstract'])) > 500 else selected_doc['abstract'])
    
    # Find similar documents
    if st.button("🔍 Find Similar Documents", key="find_similar"):
        with st.spinner("Finding similar documents..."):
            try:
                similar_docs = analyzer.find_similar_documents(selected_doc_idx, top_n=num_similar)
                
                if len(similar_docs) > 0:
                    st.success(f"Found {len(similar_docs)} similar documents")
                    
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
                        
                        with st.expander(f"{icon} #{rank} - Similarity: {similarity:.3f} - {row['title'][:70]}..."):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Title**: {row['title']}")
                                if 'authors' in row:
                                    st.markdown(f"**Authors**: {row['authors']}")
                                if 'year' in row and pd.notna(row['year']):
                                    st.markdown(f"**Year**: {int(row['year'])}")
                                st.markdown("---")
                                abstract = str(row['abstract'])[:400]
                                st.write(abstract + "..." if len(str(row['abstract'])) > 400 else row['abstract'])
                            
                            with col2:
                                st.metric("Similarity", f"{similarity:.3f}")
                                if 'dominant_topic' in row:
                                    st.metric("Dominant Topic", f"Topic {int(row['dominant_topic'])}")
                else:
                    st.info("No similar documents found")
            except Exception as e:
                st.error(f"Error finding similar documents: {e}")

with tab2:
    st.subheader("Analyze Topic Distribution of New Text")
    st.markdown("Enter text below to see its topic distribution based on the trained model.")
    
    # Text input
    new_text = st.text_area(
        "Enter text to analyze",
        height=200,
        placeholder="Paste or type your text here...",
        key="new_text_input",
    )
    
    if st.button("🔍 Analyze Text", key="analyze_text"):
        if not new_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                try:
                    # Infer topics
                    result = model.infer_topics(
                        new_text,
                        preprocessor=preprocessor,
                        num_words=10,
                    )
                    
                    st.success("Analysis complete!")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### Preprocessing Info")
                        st.metric("Tokens", result['num_tokens'])
                        
                        with st.expander("View tokens"):
                            st.write(result['tokens'][:50])
                            if len(result['tokens']) > 50:
                                st.caption(f"... and {len(result['tokens']) - 50} more")
                    
                    with col2:
                        st.markdown("### Top Topics")
                        
                        for topic_info in result['topics'][:5]:
                            topic_id = topic_info['topic_id']
                            prob = topic_info['probability']
                            words = ", ".join(topic_info['top_words'][:5])
                            
                            st.progress(
                                prob,
                                text=f"Topic {topic_id}: {prob:.1%}",
                            )
                            st.caption(f"Words: {words}")
                    
                    # Find similar based on this text
                    st.markdown("---")
                    st.markdown("### Similar Documents")
                    
                    try:
                        similar_docs = analyzer.find_similar_by_text(
                            new_text,
                            preprocessor=preprocessor,
                            top_n=5,
                        )
                        
                        if len(similar_docs) > 0:
                            for rank, (_, row) in enumerate(similar_docs.iterrows(), 1):
                                similarity = row['similarity']
                                st.markdown(f"**{rank}. (Sim: {similarity:.3f})** {row['title'][:80]}...")
                        else:
                            st.info("No similar documents found")
                    except Exception as e:
                        st.warning(f"Could not find similar documents: {e}")
                        
                except Exception as e:
                    st.error(f"Error analyzing text: {e}")

# Sidebar with tips
with st.sidebar:
    st.header("💡 Tips")
    
    st.markdown("""
    **Finding Similar Documents:**
    - Select a document from the dropdown
    - The search uses topic distribution similarity
    - Higher similarity scores mean more similar topics
    
    **Analyzing New Text:**
    - Enter or paste any text
    - Works best with academic abstracts
    - Longer text gives better results
    - Uses the same preprocessing as training
    """)
