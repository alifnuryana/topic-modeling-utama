"""
Fungsi utilitas dashboard.

Menyediakan utilitas caching dan loading untuk dashboard Streamlit.
"""

import json
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


class TopicLabelManager:
    """
    Manager untuk label topik kustom.
    
    Menyimpan dan memuat label topik dari file JSON terpisah,
    memungkinkan user untuk memberikan nama yang lebih deskriptif
    untuk setiap topik tanpa perlu melatih ulang model.
    """
    
    LABELS_FILENAME = "topic_labels.json"
    
    def __init__(self) -> None:
        """Initialize TopicLabelManager."""
        self.settings = get_settings()
        self._labels: dict[int, str] = {}
        self._loaded = False
    
    @property
    def labels_file_path(self) -> Path:
        """Path ke file labels JSON."""
        return self.settings.models_dir / self.LABELS_FILENAME
    
    def has_labels(self) -> bool:
        """Cek apakah file label kustom ada dan valid."""
        if not self.labels_file_path.exists():
            return False
        try:
            with open(self.labels_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return isinstance(data, dict) and len(data) > 0
        except (json.JSONDecodeError, IOError):
            return False
    
    def load_labels(self) -> dict[int, str]:
        """Memuat label dari file JSON."""
        if self._loaded:
            return self._labels
        
        if self.labels_file_path.exists():
            try:
                with open(self.labels_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Convert string keys to int
                self._labels = {int(k): v for k, v in data.items()}
            except (json.JSONDecodeError, IOError):
                self._labels = {}
        else:
            self._labels = {}
        
        self._loaded = True
        return self._labels
    
    def save_labels(self, labels: dict[int, str]) -> bool:
        """
        Menyimpan label ke file JSON.
        
        Args:
            labels: Dict mapping topic_id -> label
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            self.settings.models_dir.mkdir(parents=True, exist_ok=True)
            with open(self.labels_file_path, "w", encoding="utf-8") as f:
                json.dump(labels, f, indent=2, ensure_ascii=False)
            self._labels = labels
            self._loaded = True
            return True
        except IOError:
            return False
    
    def get_label(self, topic_id: int, default: Optional[str] = None) -> str:
        """
        Mendapatkan label untuk topik tertentu.
        
        Args:
            topic_id: ID topik
            default: Label default jika tidak ada
            
        Returns:
            Label topik atau default
        """
        labels = self.load_labels()
        if topic_id in labels:
            return labels[topic_id]
        return default or f"Topik {topic_id}"
    
    def get_all_labels(self) -> dict[int, str]:
        """Mendapatkan semua label."""
        return self.load_labels()
    
    def set_label(self, topic_id: int, label: str) -> bool:
        """
        Mengatur label untuk satu topik.
        
        Args:
            topic_id: ID topik
            label: Label baru
            
        Returns:
            True jika berhasil
        """
        labels = self.load_labels()
        labels[topic_id] = label
        return self.save_labels(labels)
    
    def reset_labels(self) -> bool:
        """
        Menghapus file label (reset ke default).
        
        Returns:
            True jika berhasil
        """
        try:
            if self.labels_file_path.exists():
                self.labels_file_path.unlink()
            self._labels = {}
            self._loaded = False
            return True
        except IOError:
            return False
    
    def get_labels_with_defaults(self, model: "LDATopicModel") -> dict[int, str]:
        """
        Mendapatkan semua label dengan fallback ke label otomatis dari model.
        
        Args:
            model: LDATopicModel instance
            
        Returns:
            Dict mapping topic_id -> label (kustom atau auto)
        """
        custom_labels = self.load_labels()
        result = {}
        
        topics = model.get_topics()
        for topic in topics:
            if topic.topic_id in custom_labels:
                result[topic.topic_id] = custom_labels[topic.topic_id]
            else:
                # Fallback ke auto-generated label (top 3 words)
                top_words = ", ".join(topic.top_words[:3])
                result[topic.topic_id] = f"{top_words}"
        
        return result


@st.cache_resource
def get_topic_label_manager() -> TopicLabelManager:
    """Mendapatkan instance TopicLabelManager dengan caching."""
    return TopicLabelManager()


@st.cache_resource
def load_model() -> Optional[LDATopicModel]:
    """Memuat model LDA yang telah dilatih dengan caching."""
    try:
        settings = get_settings()
        model = LDATopicModel(settings)
        model.load()
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None


@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """Memuat matriks topik-dokumen dengan caching."""
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
        st.error(f"Error memuat data: {e}")
        return None


@st.cache_data
def load_processed_docs() -> Optional[list]:
    """Memuat dokumen terproses dengan caching."""
    try:
        settings = get_settings()
        corpus_path = settings.processed_data_dir / settings.processed_corpus_file
        
        if corpus_path.exists():
            with open(corpus_path, 'rb') as f:
                corpus_data = pickle.load(f)
            return corpus_data.get('documents')
        
        return None
    except Exception as e:
        st.error(f"Error memuat dokumen: {e}")
        return None


@st.cache_resource
def load_preprocessor() -> Optional[IndonesianPreprocessor]:
    """Memuat preprocessor dengan caching."""
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
        st.error(f"Error memuat preprocessor: {e}")
        return None


@st.cache_resource
def get_analyzer(_model: LDATopicModel) -> Optional[TopicAnalyzer]:
    """Mendapatkan instance topic analyzer."""
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
        st.error(f"Error membuat analyzer: {e}")
        return None


@st.cache_resource
def get_visualizer(_model: LDATopicModel) -> TopicVisualizer:
    """Mendapatkan instance topic visualizer."""
    settings = get_settings()
    return TopicVisualizer(_model, settings)


def get_topic_colors(num_topics: int) -> list:
    """Mendapatkan daftar warna untuk topik."""
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
    """Format label topik dari kata-kata teratasnya."""
    word_str = ", ".join(words[:max_words])
    return f"Topik {topic_id}: {word_str}"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Memotong teks dengan elipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def check_model_loaded() -> bool:
    """Memeriksa apakah model sudah dimuat dan menampilkan pesan jika belum."""
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Model belum dimuat. Silakan jalankan notebook terlebih dahulu untuk melatih model.")
        st.info("""
        **Untuk melatih model:**
        1. Jalankan `01_data_collection.ipynb` untuk mengumpulkan data
        2. Jalankan `02_data_cleaning.ipynb` untuk membersihkan data
        3. Jalankan `03_preprocessing.ipynb` untuk preprocessing teks
        4. Jalankan `04_lda_modeling.ipynb` untuk melatih model LDA
        """)
        return False
    
    return True
