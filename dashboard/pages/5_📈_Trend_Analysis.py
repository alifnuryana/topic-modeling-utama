"""
Halaman Analisis Tren untuk Dashboard Pemodelan Topik.
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

from dashboard.utils import load_model, load_data, get_analyzer, check_model_loaded
from dashboard.components.charts import create_topic_trend_chart
from dashboard.components.filters import multi_topic_selector, date_range_selector

st.set_page_config(
    page_title="Analisis Tren - Pemodelan Topik",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Analisis Tren")
st.markdown("Analisis bagaimana topik berubah dari waktu ke waktu.")
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

# Check if year column exists
if 'year' not in df.columns or df['year'].isna().all():
    st.warning("⚠️ Data temporal tidak tersedia. Dataset tidak memiliki informasi tahun yang valid.")
    st.info("Analisis tren memerlukan dokumen dengan tanggal/tahun.")
    st.stop()

# Get year range
years = df['year'].dropna().astype(int)
min_year = int(years.min())
max_year = int(years.max())

# Sidebar
with st.sidebar:
    st.header("Opsi")
    
    # Topic selection
    selected_topics = multi_topic_selector(
        num_topics,
        key="trend_topics",
        label="Topik yang Ditampilkan",
        default=list(range(min(5, num_topics))),
    )
    
    st.markdown("---")
    
    # Year range
    year_range = date_range_selector(
        min_year,
        max_year,
        key="trend_years",
        label="Rentang Tahun",
    )
    
    st.markdown("---")
    
    # Aggregation
    agg_method = st.selectbox(
        "Metode Agregasi",
        options=["Rata-rata", "Jumlah", "Hitung"],
        index=0,
        key="agg_method",
    )

# Filter data by year
filtered_df = df[
    (df['year'] >= year_range[0]) & 
    (df['year'] <= year_range[1])
].copy()

st.subheader(f"Tren Topik ({year_range[0]} - {year_range[1]})")
st.caption(f"Menganalisis {len(filtered_df):,} dokumen")

# Compute trends
if len(selected_topics) == 0:
    st.warning("Silakan pilih setidaknya satu topik untuk ditampilkan.")
else:
    try:
        # Get topic columns
        topic_columns = [f'topic_{t}' for t in selected_topics if f'topic_{t}' in filtered_df.columns]
        
        if not topic_columns:
            st.error("Kolom topik tidak ditemukan dalam data.")
        else:
            # Group by year
            filtered_df['year_int'] = filtered_df['year'].astype(int)
            
            if agg_method == "Rata-rata":
                trends = filtered_df.groupby('year_int')[topic_columns].mean().reset_index()
            elif agg_method == "Jumlah":
                trends = filtered_df.groupby('year_int')[topic_columns].sum().reset_index()
            else:  # Hitung
                # Count documents where topic is dominant
                count_data = []
                for year in filtered_df['year_int'].unique():
                    year_df = filtered_df[filtered_df['year_int'] == year]
                    row = {'year_int': year}
                    for t in selected_topics:
                        col = f'topic_{t}'
                        if col in year_df.columns:
                            row[col] = (year_df[col] > 0.3).sum()
                    count_data.append(row)
                trends = pd.DataFrame(count_data)
            
            trends = trends.sort_values('year_int')
            trends['_date'] = pd.to_datetime(trends['year_int'], format='%Y')
            
            # Create trend chart
            fig = create_topic_trend_chart(
                trends,
                selected_topics=selected_topics,
                date_column='_date',
                height=500,
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Year-by-year comparison
            st.markdown("---")
            st.subheader("📊 Statistik Tahun demi Tahun")
            
            # Create comparison table
            comparison_data = []
            for _, row in trends.iterrows():
                year_data = {'Tahun': int(row['year_int'])}
                for col in topic_columns:
                    topic_id = col.split('_')[1]
                    year_data[f'Topik {topic_id}'] = f"{row[col]:.4f}" if agg_method != "Hitung" else int(row[col])
                comparison_data.append(year_data)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv_data = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Unduh Data Tren (CSV)",
                data=csv_data,
                file_name="tren_topik.csv",
                mime="text/csv",
            )
            
    except Exception as e:
        st.error(f"Error menghitung tren: {e}")
        import traceback
        st.code(traceback.format_exc())

# Topic prevalence by year
st.markdown("---")
st.subheader("📊 Prevalensi Topik berdasarkan Periode")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Periode Awal** (Paruh pertama)")
    mid_year = (min_year + max_year) // 2
    early_df = df[(df['year'] >= min_year) & (df['year'] < mid_year)]
    
    if len(early_df) > 0:
        for t in selected_topics[:5]:
            col = f'topic_{t}'
            if col in early_df.columns:
                mean_prob = early_df[col].mean()
                st.progress(mean_prob, text=f"Topik {t}: {mean_prob:.3f}")

with col2:
    st.markdown("**Periode Terkini** (Paruh kedua)")
    recent_df = df[(df['year'] >= mid_year) & (df['year'] <= max_year)]
    
    if len(recent_df) > 0:
        for t in selected_topics[:5]:
            col = f'topic_{t}'
            if col in recent_df.columns:
                mean_prob = recent_df[col].mean()
                st.progress(mean_prob, text=f"Topik {t}: {mean_prob:.3f}")

# Insights
st.markdown("---")
st.subheader("💡 Wawasan")

try:
    # Find trending topics
    if len(trends) > 1:
        growth_rates = {}
        for col in topic_columns:
            first_val = trends[col].iloc[:3].mean()  # First 3 years
            last_val = trends[col].iloc[-3:].mean()  # Last 3 years
            
            if first_val > 0:
                growth = (last_val - first_val) / first_val * 100
            else:
                growth = 0
            
            topic_id = col.split('_')[1]
            growth_rates[f"Topik {topic_id}"] = growth
        
        # Sort by growth
        sorted_growth = sorted(growth_rates.items(), key=lambda x: -x[1])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Topik Meningkat**")
            for topic, growth in sorted_growth[:3]:
                if growth > 0:
                    st.success(f"{topic}: +{growth:.1f}%")
        
        with col2:
            st.markdown("**📉 Topik Menurun**")
            for topic, growth in sorted_growth[-3:]:
                if growth < 0:
                    st.error(f"{topic}: {growth:.1f}%")
                    
except Exception as e:
    st.info("Data tidak cukup untuk menghitung tren pertumbuhan.")
