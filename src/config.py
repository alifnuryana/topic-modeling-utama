"""
Configuration module for Topic Modeling project.

Centralized configuration using Pydantic for type-safe settings management.
"""

from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # ==========================================================================
    # Project Paths
    # ==========================================================================
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Root directory of the project"
    )
    
    @property
    def data_dir(self) -> Path:
        """Directory for all data files."""
        return self.project_root / "data"
    
    @property
    def raw_data_dir(self) -> Path:
        """Directory for raw harvested data."""
        return self.data_dir / "raw"
    
    @property
    def processed_data_dir(self) -> Path:
        """Directory for cleaned and processed data."""
        return self.data_dir / "processed"
    
    @property
    def models_dir(self) -> Path:
        """Directory for trained models."""
        return self.project_root / "models"
    
    @property
    def outputs_dir(self) -> Path:
        """Directory for generated outputs."""
        return self.project_root / "outputs"
    
    # ==========================================================================
    # OAI-PMH Settings
    # ==========================================================================
    oaipmh_endpoint: str = Field(
        default="https://repository.widyatama.ac.id/oai/request",
        description="OAI-PMH endpoint URL"
    )
    oaipmh_metadata_prefix: str = Field(
        default="oai_dc",
        description="Metadata format prefix (Dublin Core)"
    )
    oaipmh_set_spec: Optional[str] = Field(
        default="com_123456789_15",
        description="Set specification for harvesting (if any)"
    )
    oaipmh_batch_size: int = Field(
        default=100,
        description="Number of records per batch"
    )
    oaipmh_delay_seconds: float = Field(
        default=0,
        description="Delay between requests to avoid rate limiting"
    )
    oaipmh_max_records: Optional[int] = Field(
        default=None,
        description="Maximum number of records to harvest (None for all)"
    )
    
    # ==========================================================================
    # Preprocessing Settings
    # ==========================================================================
    min_word_length: int = Field(
        default=3,
        description="Minimum word length to keep after preprocessing"
    )
    max_word_length: int = Field(
        default=50,
        description="Maximum word length to keep after preprocessing"
    )
    min_doc_length: int = Field(
        default=30,
        description="Minimum document length (in tokens) to include"
    )
    use_stemming: bool = Field(
        default=False,
        description="Whether to apply stemming (PySastrawi)"
    )
    use_bigrams: bool = Field(
        default=True,
        description="Whether to detect and use bigrams"
    )
    use_trigrams: bool = Field(
        default=False,
        description="Whether to detect and use trigrams"
    )
    bigram_min_count: int = Field(
        default=20,
        description="Minimum count for bigram detection"
    )
    trigram_min_count: int = Field(
        default=10,
        description="Minimum count for trigram detection"
    )
    
    # ==========================================================================
    # LDA Model Settings
    # ==========================================================================
    lda_num_topics: int = Field(
        default=0,
        description="Number of topics for LDA model"
    )
    lda_passes: int = Field(
        default=20,
        description="Number of passes through the corpus during training"
    )
    lda_iterations: int = Field(
        default=400,
        description="Maximum number of iterations per document"
    )
    lda_chunksize: int = Field(
        default=3000,
        description="Number of documents per training chunk"
    )
    lda_alpha: str = Field(
        default="symmetric",
        description="Alpha parameter for topic-document distribution"
    )
    lda_eta: Optional[str] = Field(
        default=None,
        description="Eta parameter for word-topic distribution"
    )
    lda_random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    lda_workers: int = Field(
        default=6,
        description="Number of worker threads for training"
    )
    
    # ==========================================================================
    # Dictionary Settings
    # ==========================================================================
    dict_no_below: int = Field(
        default=20,
        description="Minimum document frequency for words"
    )
    dict_no_above: float = Field(
        default=0.5,
        description="Maximum document frequency fraction for words"
    )
    dict_keep_n: int = Field(
        default=100000,
        description="Maximum vocabulary size"
    )
    
    # ==========================================================================
    # File Names
    # ==========================================================================
    raw_metadata_file: str = Field(
        default="raw_metadata.csv",
        description="Filename for raw harvested metadata"
    )
    clean_metadata_file: str = Field(
        default="clean_metadata.csv",
        description="Filename for cleaned metadata"
    )
    processed_corpus_file: str = Field(
        default="processed_corpus.pkl",
        description="Filename for processed corpus"
    )
    dictionary_file: str = Field(
        default="dictionary.pkl",
        description="Filename for gensim dictionary"
    )
    corpus_file: str = Field(
        default="corpus.mm",
        description="Filename for gensim corpus (Market Matrix)"
    )
    lda_model_file: str = Field(
        default="lda_model",
        description="Base filename for LDA model files"
    )
    topic_document_matrix_file: str = Field(
        default="topic_document_matrix.csv",
        description="Filename for topic-document distribution matrix"
    )
    
    class Config:
        env_prefix = "TM_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def ensure_directories(settings: Optional[Settings] = None) -> None:
    """Create all required directories if they don't exist."""
    if settings is None:
        settings = get_settings()
    
    directories = [
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.models_dir,
        settings.outputs_dir,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
