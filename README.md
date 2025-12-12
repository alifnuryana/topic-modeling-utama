# Topic Modeling with LDA

A comprehensive Topic Modeling system using Latent Dirichlet Allocation (LDA) for analyzing academic paper metadata from the Widyatama University repository.

## 🎯 Features

- **Data Collection**: OAI-PMH harvesting with cloudscraper protection
- **Indonesian NLP**: PySastrawi stemming, Indonesian stopwords
- **Topic Modeling**: LDA with coherence-based optimization
- **Interactive Dashboard**: Streamlit multi-page exploration
- **Comprehensive Analysis**: Trend analysis, document similarity, topic comparison

## 📁 Project Structure

```
topic-modeling-utama/
├── src/                      # Core Python modules
│   ├── config.py             # Configuration settings
│   ├── harvester.py          # OAI-PMH data collection
│   ├── preprocessor.py       # Indonesian text preprocessing
│   ├── lda_model.py          # LDA model training
│   ├── analysis.py           # Analysis utilities
│   └── visualizations.py     # Visualization functions
├── notebooks/                # Jupyter notebooks (pipeline)
│   ├── 01_data_collection.ipynb
│   ├── 01b_eda_raw_data.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 02b_eda_clean_data.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_lda_modeling.ipynb
│   └── 05_analysis_visualization.ipynb
├── dashboard/                # Streamlit dashboard
│   ├── app.py                # Entry point
│   ├── utils.py              # Utilities
│   ├── pages/                # Dashboard pages
│   └── components/           # Reusable UI components
├── data/
│   ├── raw/                  # Raw harvested data
│   └── processed/            # Cleaned and processed data
├── models/                   # Trained LDA models
└── outputs/                  # Generated visualizations
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Run the Pipeline

Execute notebooks in order:

1. **Data Collection** - `01_data_collection.ipynb`
   - Harvests metadata from repository

2. **EDA (Raw)** - `01b_eda_raw_data.ipynb`
   - Explores raw data quality

3. **Data Cleaning** - `02_data_cleaning.ipynb`
   - Cleans and validates data

4. **EDA (Clean)** - `02b_eda_clean_data.ipynb`
   - Analyzes cleaned data

5. **Preprocessing** - `03_preprocessing.ipynb`
   - Tokenization, stemming, phrase detection

6. **LDA Modeling** - `04_lda_modeling.ipynb`
   - Trains and optimizes LDA model

7. **Visualization** - `05_analysis_visualization.ipynb`
   - Generates all visualizations

### 3. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Overview and quick stats |
| 📊 **Topic Explorer** | Word clouds, top words, pyLDAvis |
| 📄 **Document Browser** | Search and filter documents |
| 🔍 **Similarity Search** | Find similar documents |
| 📈 **Trend Analysis** | Topic evolution over time |
| 🎯 **Topic Comparison** | Compare topics side-by-side |
| ⚙️ **Model Insights** | Model metrics and configuration |

## ⚙️ Configuration

Configuration is managed through `src/config.py`. Key settings:

```python
# OAI-PMH
oaipmh_endpoint = "https://repository.widyatama.ac.id/oai/request"

# LDA Model
lda_num_topics = 10
lda_passes = 15
lda_iterations = 400

# Preprocessing
use_stemming = True  # PySastrawi
use_bigrams = True
use_trigrams = True
```

Override via environment variables with `TM_` prefix:
```bash
export TM_LDA_NUM_TOPICS=15
```

## 📦 Dependencies

- **Data**: pandas, numpy, sickle, cloudscraper
- **NLP**: gensim, nltk, PySastrawi, nlp-id
- **Visualization**: matplotlib, seaborn, plotly, wordcloud, pyLDAvis
- **Dashboard**: streamlit, streamlit-option-menu
- **Notebooks**: jupyter, ipywidgets
- **Config**: pydantic, pydantic-settings

## 📝 License

MIT License
