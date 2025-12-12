"""
Document Browser page for the Topic Modeling Dashboard.
"""

import streamlit as st
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dashboard.utils import load_model, load_data, get_analyzer, check_model_loaded
from dashboard.components.filters import (
    topic_selector, search_input, probability_slider, num_results_selector
)
from dashboard.components.charts import create_topic_distribution_pie

st.set_page_config(
    page_title="Document Browser - Topic Modeling",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Browser")
st.markdown("Search and filter documents by topic or keyword.")
st.markdown("---")

# Load model
if not check_model_loaded():
    st.stop()

model = load_model()
df = load_data()
analyzer = get_analyzer(model)

if df is None:
    st.error("Could not load document data.")
    st.stop()

num_topics = model.model.num_topics

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    
    # Search
    search_query = search_input(
        key="doc_search",
        label="Search in Title/Abstract",
        placeholder="Enter keywords...",
    )
    
    st.markdown("---")
    
    # Topic filter
    filter_topic = topic_selector(
        num_topics,
        key="doc_filter_topic",
        label="Filter by Topic",
        include_all=True,
    )
    
    # Probability threshold
    if filter_topic is not None:
        min_prob = probability_slider(
            key="doc_min_prob",
            label="Min Topic Probability",
            default=0.3,
        )
    else:
        min_prob = 0.0
    
    st.markdown("---")
    
    # Year filter
    if 'year' in df.columns and df['year'].notna().any():
        years = df['year'].dropna().astype(int)
        year_range = st.slider(
            "Year Range",
            min_value=int(years.min()),
            max_value=int(years.max()),
            value=(int(years.min()), int(years.max())),
            key="year_filter",
        )
    else:
        year_range = None
    
    st.markdown("---")
    
    # Results per page
    results_per_page = num_results_selector(
        key="results_per_page",
        label="Results per Page",
        options=[10, 20, 50, 100],
        default_index=1,
    )

# Apply filters
filtered_df = df.copy()

# Search filter
if search_query:
    search_lower = search_query.lower()
    mask = (
        filtered_df['title'].str.lower().str.contains(search_lower, na=False) |
        filtered_df['abstract'].str.lower().str.contains(search_lower, na=False)
    )
    filtered_df = filtered_df[mask]

# Topic filter
if filter_topic is not None and f'topic_{filter_topic}' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df[f'topic_{filter_topic}'] >= min_prob]
    filtered_df = filtered_df.sort_values(f'topic_{filter_topic}', ascending=False)

# Year filter
if year_range and 'year' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['year'] >= year_range[0]) & 
        (filtered_df['year'] <= year_range[1])
    ]

# Display results
st.subheader(f"📚 Results ({len(filtered_df):,} documents)")

if len(filtered_df) == 0:
    st.info("No documents match your filters. Try adjusting the criteria.")
else:
    # Pagination
    total_pages = (len(filtered_df) - 1) // results_per_page + 1
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            key="page_number",
        )
    
    start_idx = (page - 1) * results_per_page
    end_idx = start_idx + results_per_page
    
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    st.caption(f"Showing {start_idx + 1} - {min(end_idx, len(filtered_df))} of {len(filtered_df)}")
    
    # Display documents
    for idx, (_, row) in enumerate(page_df.iterrows()):
        with st.expander(
            f"📄 {row['title'][:100]}..." if len(str(row['title'])) > 100 else f"📄 {row['title']}",
            expanded=idx == 0,
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Metadata
                if 'authors' in row and row['authors']:
                    st.markdown(f"**Authors**: {row['authors']}")
                
                if 'year' in row and pd.notna(row['year']):
                    st.markdown(f"**Year**: {int(row['year'])}")
                
                if 'subjects' in row and row['subjects']:
                    subjects = str(row['subjects'])[:200]
                    st.markdown(f"**Subjects**: {subjects}...")
                
                st.markdown("---")
                
                # Abstract
                st.markdown("**Abstract**:")
                abstract = str(row['abstract'])
                if len(abstract) > 500:
                    st.write(abstract[:500] + "...")
                    with st.expander("Show full abstract"):
                        st.write(abstract)
                else:
                    st.write(abstract)
            
            with col2:
                # Topic distribution
                st.markdown("**Topic Distribution**:")
                
                if 'dominant_topic' in row:
                    st.metric(
                        "Dominant Topic",
                        f"Topic {int(row['dominant_topic'])}",
                        f"{row.get('dominant_prob', 0):.1%}" if 'dominant_prob' in row else None,
                    )
                
                # Get topic probabilities
                topic_probs = []
                for t in range(num_topics):
                    col_name = f'topic_{t}'
                    if col_name in row:
                        prob = row[col_name]
                        if prob > 0.05:  # Only show significant topics
                            topic_probs.append((t, prob))
                
                topic_probs.sort(key=lambda x: -x[1])
                
                if topic_probs:
                    # Simple bar display
                    for topic_id, prob in topic_probs[:5]:
                        st.progress(
                            prob,
                            text=f"Topic {topic_id}: {prob:.1%}",
                        )

# Download option
st.markdown("---")

if len(filtered_df) > 0:
    st.download_button(
        label="📥 Download Filtered Results (CSV)",
        data=filtered_df.drop(columns=[c for c in filtered_df.columns if c.startswith('topic_')], errors='ignore').to_csv(index=False),
        file_name="filtered_documents.csv",
        mime="text/csv",
    )
