"""
Indonesian Text Preprocessor module.

Provides text preprocessing pipeline optimized for Indonesian language
with support for multilingual text.
"""

import re
import time
import logging
import pickle
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nlp_id.stopword import StopWord
from nlp_id.tokenizer import Tokenizer
from gensim.models.phrases import Phrases, Phraser
# from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
import nltk

from src.config import Settings, get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Ensure NLTK data is downloaded
def ensure_nltk_data() -> None:
    """Download required NLTK data if not present."""
    try:
        nltk.data.find('stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)


@dataclass
class PreprocessingStats:
    """Statistics from preprocessing operation."""
    
    total_documents: int = 0
    processed_documents: int = 0
    skipped_documents: int = 0
    total_tokens: int = 0
    unique_tokens: int = 0
    avg_tokens_per_doc: float = 0.0
    stemming_time_seconds: float = 0.0
    total_time_seconds: float = 0.0


class IndonesianPreprocessor:
    """
    Text preprocessing pipeline for Indonesian language.
    
    Features:
    - Case normalization
    - Punctuation and number removal
    - Tokenization
    - Stopword removal (Indonesian + custom)
    - Stemming with PySastrawi (with performance tracking)
    - Bigram/Trigram phrase detection
    """
    
    # Default Indonesian stopwords to add beyond Sastrawi's
    ADDITIONAL_STOPWORDS: set[str] = {
        # Common Indonesian words
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan",
        "untuk", "pada", "adalah", "dalam", "tidak", "akan", "juga",
        "atau", "ada", "mereka", "sudah", "saya", "kami", "kita",
        "bisa", "dapat", "harus", "seperti", "oleh", "karena",
        "sebuah", "satu", "dua", "tiga", "tersebut", "serta",
        "namun", "tetapi", "bahwa", "lebih", "telah", "sangat",
        "secara", "antara", "melalui", "setelah", "semua", "hanya",
        # Academic/thesis common words
        "penelitian", "hasil", "metode", "data", "analisis",
        "menggunakan", "berdasarkan", "menunjukkan", "dilakukan",
        "digunakan", "diperoleh", "mengetahui", "bertujuan",
        "kesimpulan", "saran", "pembahasan", "pendahuluan",
        # English common words (for multilingual support)
        "the", "and", "of", "to", "in", "is", "for", "on", "with",
        "that", "this", "are", "as", "be", "by", "an", "was", "at",
        "or", "from", "it", "has", "have", "been", "were", "which",
        "can", "will", "also", "more", "its", "their", "all",
    }
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        custom_stopwords: Optional[set[str]] = None,
        use_stemming: Optional[bool] = None,
        use_bigrams: Optional[bool] = None,
        use_trigrams: Optional[bool] = None,
    ) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            settings: Project settings
            custom_stopwords: Additional stopwords to use
            use_stemming: Override stemming setting
            use_bigrams: Override bigram detection setting
            use_trigrams: Override trigram detection setting
        """
        ensure_nltk_data()
        
        self.settings = settings or get_settings()
        
        # Settings with overrides
        self._use_stemming = use_stemming if use_stemming is not None else self.settings.use_stemming
        self._use_bigrams = use_bigrams if use_bigrams is not None else self.settings.use_bigrams
        self._use_trigrams = use_trigrams if use_trigrams is not None else self.settings.use_trigrams
        
        # Initialize Sastrawi components
        self._init_sastrawi()

        # Initialize NLP-ID components
        self._init_nlp_id()

        # Initialize NLTK components
        self._init_nltk()
        
        # Build stopwords set
        self.stopwords = self._build_stopwords(custom_stopwords)
        
        # Phrase models (initialized during fit)
        self.bigram_model: Optional[Phraser] = None
        self.trigram_model: Optional[Phraser] = None
        
        # Statistics
        self.stats = PreprocessingStats()
        
        logger.info(f"Preprocessor initialized (stemming={self._use_stemming})")

    def _init_nltk(self):
        self._nltk_stopwords = set(nltk_stopwords.words('english'))

    def _init_nlp_id(self) -> None:
        # Get default stopwords from nlp-id
        stopword = StopWord()
        self._nlp_id_stopwords = set(stopword.get_stopword())

        self._nlp_id_tokenizer = Tokenizer()

    def _init_sastrawi(self) -> None:
        """Initialize Sastrawi stemmer and stopword remover."""
        # Stemmer
        stemmer_factory = StemmerFactory()
        self._stemmer = stemmer_factory.create_stemmer()

    
    def _build_stopwords(self, custom_stopwords: Optional[set[str]] = None) -> set[str]:
        """Build complete stopwords set."""
        stopwords = set()
        # stopwords.update(self._sastrawi_stopwords)
        stopwords.update(self._nltk_stopwords)
        stopwords.update(self._nlp_id_stopwords)
        stopwords.update(self.ADDITIONAL_STOPWORDS)
        
        if custom_stopwords:
            stopwords.update(custom_stopwords)
        
        logger.info(f"Built stopwords set with {len(stopwords)} words")
        return stopwords
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation and special characters, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Cleaned text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        try:
            tokens = self._nlp_id_tokenizer.tokenize(text)
        except Exception:
            # Fallback to simple split
            tokens = text.split()
        
        return tokens
    
    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered token list
        """
        return [t for t in tokens if t not in self.stopwords]
    
    def filter_by_length(self, tokens: list[str]) -> list[str]:
        """
        Filter tokens by length.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered token list
        """
        return [
            t for t in tokens
            if self.settings.min_word_length <= len(t) <= self.settings.max_word_length
        ]
    
    def stem_tokens(
        self,
        tokens: list[str],
        show_progress: bool = False,
    ) -> list[str]:
        """
        Stem tokens using PySastrawi.
        
        Args:
            tokens: List of tokens
            show_progress: Show progress for large token lists
            
        Returns:
            List of stemmed tokens
        """
        if not self._use_stemming:
            return tokens
        
        start_time = time.time()
        
        stemmed = []
        token_iter = tqdm(tokens, desc="Stemming", disable=not show_progress)
        
        for token in token_iter:
            try:
                stemmed.append(self._stemmer.stem(token))
            except Exception:
                stemmed.append(token)
        
        elapsed = time.time() - start_time
        self.stats.stemming_time_seconds += elapsed
        
        if show_progress:
            logger.info(f"Stemmed {len(tokens)} tokens in {elapsed:.2f}s")
        
        return stemmed
    
    def preprocess_text(
        self,
        text: str,
        apply_phrases: bool = True,
    ) -> list[str]:
        """
        Full preprocessing pipeline for a single text.
        
        Args:
            text: Raw input text
            apply_phrases: Apply bigram/trigram models if available
            
        Returns:
            List of preprocessed tokens
        """
        # Clean
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)

        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Filter by length
        tokens = self.filter_by_length(tokens)
        
        # Stem
        if self._use_stemming:
            tokens = self.stem_tokens(tokens)
        
        # Apply phrase models
        if apply_phrases and self.bigram_model:
            tokens = list(self.bigram_model[tokens])
            
            if self.trigram_model:
                tokens = list(self.trigram_model[tokens])
        
        return tokens
    
    def fit_phrase_models(
        self,
        tokenized_docs: list[list[str]],
    ) -> None:
        """
        Fit bigram and trigram phrase models.
        
        Args:
            tokenized_docs: List of tokenized documents
        """
        if self._use_bigrams:
            logger.info("Fitting bigram model...")
            bigram_phrases = Phrases(
                tokenized_docs,
                min_count=self.settings.bigram_min_count,
                threshold=10,
            )
            self.bigram_model = Phraser(bigram_phrases)
            
            if self._use_trigrams:
                logger.info("Fitting trigram model...")
                # Apply bigrams first
                bigram_docs = [self.bigram_model[doc] for doc in tokenized_docs]
                trigram_phrases = Phrases(
                    bigram_docs,
                    min_count=self.settings.trigram_min_count,
                    threshold=10,
                )
                self.trigram_model = Phraser(trigram_phrases)
    
    def preprocess_documents(
        self,
        documents: list[str],
        fit_phrases: bool = True,
        show_progress: bool = True,
    ) -> list[list[str]]:
        """
        Preprocess a list of documents.
        
        Args:
            documents: List of raw text documents
            fit_phrases: Fit phrase models on this corpus
            show_progress: Show progress bar
            
        Returns:
            List of tokenized, preprocessed documents
        """
        start_time = time.time()
        self.stats = PreprocessingStats(total_documents=len(documents))
        
        logger.info(f"Preprocessing {len(documents)} documents...")
        
        # Step 1: Basic preprocessing (no phrases yet)
        processed_docs: list[list[str]] = []
        doc_iter = tqdm(documents, desc="Preprocessing", disable=not show_progress)
        
        for doc in doc_iter:
            tokens = self.preprocess_text(doc, apply_phrases=False)
            
            if len(tokens) >= self.settings.min_doc_length:
                processed_docs.append(tokens)
                self.stats.processed_documents += 1
            else:
                self.stats.skipped_documents += 1
        
        # Step 2: Fit phrase models
        if fit_phrases and (self._use_bigrams or self._use_trigrams):
            self.fit_phrase_models(processed_docs)
            
            # Apply phrase models
            if self.bigram_model:
                processed_docs = [list(self.bigram_model[doc]) for doc in processed_docs]
                
                if self.trigram_model:
                    processed_docs = [list(self.trigram_model[doc]) for doc in processed_docs]
        
        # Calculate stats
        all_tokens = [t for doc in processed_docs for t in doc]
        self.stats.total_tokens = len(all_tokens)
        self.stats.unique_tokens = len(set(all_tokens))
        self.stats.avg_tokens_per_doc = (
            self.stats.total_tokens / self.stats.processed_documents
            if self.stats.processed_documents > 0 else 0
        )
        self.stats.total_time_seconds = time.time() - start_time
        
        logger.info(f"Preprocessing complete in {self.stats.total_time_seconds:.2f}s")
        logger.info(f"  Processed: {self.stats.processed_documents} documents")
        logger.info(f"  Skipped: {self.stats.skipped_documents} documents")
        logger.info(f"  Unique tokens: {self.stats.unique_tokens}")
        
        if self._use_stemming:
            logger.info(f"  Stemming time: {self.stats.stemming_time_seconds:.2f}s")
        
        return processed_docs
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "abstract",
        output_column: str = "tokens",
        fit_phrases: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Preprocess text column in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to preprocess
            output_column: Column name for preprocessed tokens
            fit_phrases: Fit phrase models
            show_progress: Show progress bar
            
        Returns:
            DataFrame with added tokens column
        """
        documents = df[text_column].fillna("").tolist()
        processed_docs = self.preprocess_documents(
            documents,
            fit_phrases=fit_phrases,
            show_progress=show_progress,
        )
        
        # Create new dataframe with only valid documents
        valid_indices = []
        doc_idx = 0
        
        for i, doc in enumerate(documents):
            tokens = self.preprocess_text(doc, apply_phrases=False)
            if len(tokens) >= self.settings.min_doc_length:
                valid_indices.append(i)
        
        result_df = df.iloc[valid_indices].copy()
        result_df[output_column] = processed_docs
        
        return result_df
    
    def save(self, path: Path) -> None:
        """
        Save preprocessor state (phrase models) to file.
        
        Args:
            path: Output file path
        """
        state = {
            "bigram_model": self.bigram_model,
            "trigram_model": self.trigram_model,
            "stopwords": self.stopwords,
            "stats": self.stats,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved preprocessor to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load preprocessor state from file.
        
        Args:
            path: Input file path
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.bigram_model = state.get("bigram_model")
        self.trigram_model = state.get("trigram_model")
        self.stopwords = state.get("stopwords", self.stopwords)
        self.stats = state.get("stats", PreprocessingStats())
        
        logger.info(f"Loaded preprocessor from {path}")
