"""
Halaman Peramban Dokumen untuk Dashboard Pemodelan Topik.
"""

import streamlit as st
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dashboard.utils import load_model, load_data, get_analyzer, check_model_loaded, get_topic_label_manager
from dashboard.components.filters import (
    topic_selector, search_input, probability_slider, num_results_selector
)
from dashboard.components.charts import create_topic_distribution_pie

st.set_page_config(
    page_title="Peramban Dokumen - Pemodelan Topik",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Peramban Dokumen")
st.markdown("Cari dan filter dokumen berdasarkan topik atau kata kunci.")
st.markdown("---")

# Load model
if not check_model_loaded():
    st.stop()

model = load_model()
df = load_data()
analyzer = get_analyzer(model)

if df is None:
    st.error("Tidak dapat memuat data dokumen.")
    st.stop()

num_topics = model.model.num_topics

# Load topic labels
label_manager = get_topic_label_manager()
topic_labels = label_manager.get_labels_with_defaults(model)

# Sidebar filters
with st.sidebar:
    st.header("Filter")
    
    # Search
    search_query = search_input(
        key="doc_search",
        label="Cari di Judul/Abstrak",
        placeholder="Masukkan kata kunci...",
    )
    
    st.markdown("---")
    
    # Topic filter
    filter_topic = topic_selector(
        num_topics,
        key="doc_filter_topic",
        label="Filter berdasarkan Topik",
        include_all=True,
        labels=topic_labels,
    )
    
    # Probability threshold
    if filter_topic is not None:
        min_prob = probability_slider(
            key="doc_min_prob",
            label="Probabilitas Topik Minimum",
            default=0.3,
        )
    else:
        min_prob = 0.0
    
    st.markdown("---")
    
    # Year filter
    if 'year' in df.columns and df['year'].notna().any():
        years = df['year'].dropna().astype(int)
        year_range = st.slider(
            "Rentang Tahun",
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
        label="Hasil per Halaman",
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
st.subheader(f"📚 Hasil ({len(filtered_df):,} dokumen)")

if len(filtered_df) == 0:
    st.info("Tidak ada dokumen yang cocok dengan filter Anda. Coba sesuaikan kriteria.")
else:
    # Pagination
    total_pages = (len(filtered_df) - 1) // results_per_page + 1
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input(
            "Halaman",
            min_value=1,
            max_value=total_pages,
            value=1,
            key="page_number",
        )
    
    start_idx = (page - 1) * results_per_page
    end_idx = start_idx + results_per_page
    
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    st.caption(f"Menampilkan {start_idx + 1} - {min(end_idx, len(filtered_df))} dari {len(filtered_df)}")
    
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
                    st.markdown(f"**Penulis**: {row['authors']}")
                
                if 'year' in row and pd.notna(row['year']):
                    st.markdown(f"**Tahun**: {int(row['year'])}")
                
                if 'subjects' in row and row['subjects']:
                    subjects = str(row['subjects'])[:200]
                    st.markdown(f"**Subjek**: {subjects}...")
                
                st.markdown("---")
                
                # Abstract
                st.markdown("**Abstrak**:")
                abstract = str(row['abstract'])
                if len(abstract) > 500:
                    st.write(abstract[:500] + "...")
                    with st.expander("Tampilkan abstrak lengkap"):
                        st.write(abstract)
                else:
                    st.write(abstract)
            
            with col2:
                # Topic distribution
                st.markdown("**Distribusi Topik**:")
                
                if 'dominant_topic' in row:
                    st.metric(
                        "Topik Dominan",
                        f"Topik {int(row['dominant_topic'])}",
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
                        topic_label = topic_labels.get(topic_id, f"Topik {topic_id}")
                        st.progress(
                            prob,
                            text=f"Topik {topic_id} ({topic_label}): {prob:.1%}",
                        )

# Download option
st.markdown("---")

if len(filtered_df) > 0:
    st.download_button(
        label="📥 Unduh Hasil Filter (CSV)",
        data=filtered_df.drop(columns=[c for c in filtered_df.columns if c.startswith('topic_')], errors='ignore').to_csv(index=False),
        file_name="dokumen_terfilter.csv",
        mime="text/csv",
    )
