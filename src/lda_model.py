"""
LDA Topic Model module.

Provides LDA model training, persistence, and topic extraction utilities
using Gensim's LdaMulticore for efficient parallel training.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel
from gensim.models.ldamodel import LdaModel
from tqdm import tqdm

from src.config import Settings, get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TopicInfo:
    """Information about a single topic."""
    
    topic_id: int
    words: list[tuple[str, float]]  # (word, weight) pairs
    label: str = ""
    coherence: float = 0.0
    
    @property
    def top_words(self) -> list[str]:
        """Get list of top words without weights."""
        return [word for word, _ in self.words]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic_id": self.topic_id,
            "words": self.words,
            "label": self.label,
            "coherence": self.coherence,
            "top_words": self.top_words,
        }


@dataclass
class ModelMetadata:
    """Metadata about the trained model."""
    
    num_topics: int
    num_documents: int
    vocabulary_size: int
    passes: int
    iterations: int
    alpha: str
    eta: Optional[str]
    coherence_score: float
    training_time_seconds: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "ModelMetadata":
        """Load metadata from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class LDATopicModel:
    """
    LDA Topic Model wrapper with training, evaluation, and inference.
    
    Uses Gensim's LdaMulticore for efficient parallel training.
    """
    
    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize the topic model.
        
        Args:
            settings: Project settings
        """
        self.settings = settings or get_settings()
        
        self.dictionary: Optional[corpora.Dictionary] = None
        self.corpus: Optional[list] = None
        self.model: Optional[LdaMulticore] = None
        self.metadata: Optional[ModelMetadata] = None
        self._topics: list[TopicInfo] = []
    
    def create_dictionary(
        self,
        tokenized_docs: list[list[str]],
        filter_extremes: bool = True,
    ) -> corpora.Dictionary:
        """
        Create a Gensim dictionary from tokenized documents.
        
        Args:
            tokenized_docs: List of tokenized documents
            filter_extremes: Apply frequency filtering
            
        Returns:
            Gensim Dictionary
        """
        logger.info("Creating dictionary...")
        
        self.dictionary = corpora.Dictionary(tokenized_docs)
        original_size = len(self.dictionary)
        
        if filter_extremes:
            self.dictionary.filter_extremes(
                no_below=self.settings.dict_no_below,
                no_above=self.settings.dict_no_above,
                keep_n=self.settings.dict_keep_n,
            )
        
        logger.info(f"Dictionary: {original_size} -> {len(self.dictionary)} terms")
        
        return self.dictionary
    
    def create_corpus(
        self,
        tokenized_docs: list[list[str]],
    ) -> list:
        """
        Create a bag-of-words corpus from tokenized documents.
        
        Args:
            tokenized_docs: List of tokenized documents
            
        Returns:
            Gensim corpus (list of BoW representations)
        """
        if self.dictionary is None:
            self.create_dictionary(tokenized_docs)
        
        logger.info("Creating corpus...")
        self.corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]
        
        logger.info(f"Corpus: {len(self.corpus)} documents")
        
        return self.corpus
    
    def train(
        self,
        tokenized_docs: list[list[str]],
        num_topics: Optional[int] = None,
        passes: Optional[int] = None,
        iterations: Optional[int] = None,
        alpha: Optional[str] = None,
        eta: Optional[str] = None,
        workers: Optional[int] = None,
        show_progress: bool = True,
    ) -> LdaMulticore:
        """
        Train the LDA model.
        
        Args:
            tokenized_docs: List of tokenized documents
            num_topics: Number of topics (overrides settings)
            passes: Number of passes (overrides settings)
            iterations: Max iterations per document (overrides settings)
            alpha: Alpha parameter (overrides settings)
            eta: Eta parameter (overrides settings)
            workers: Number of workers (overrides settings)
            show_progress: Show training progress
            
        Returns:
            Trained LDA model
        """
        import time
        start_time = time.time()
        
        # Use provided values or fall back to settings
        num_topics = num_topics or self.settings.lda_num_topics
        passes = passes or self.settings.lda_passes
        iterations = iterations or self.settings.lda_iterations
        alpha = alpha or self.settings.lda_alpha
        eta = eta or self.settings.lda_eta
        workers = workers or self.settings.lda_workers
        
        # Create dictionary and corpus
        self.create_dictionary(tokenized_docs)
        self.create_corpus(tokenized_docs)
        
        logger.info(f"Training LDA model with {num_topics} topics...")
        logger.info(f"  Passes: {passes}, Iterations: {iterations}")
        logger.info(f"  Alpha: {alpha}, Eta: {eta}")
        logger.info(f"  Workers: {workers}")
        
        # Train model
        self.model = LdaMulticore(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            alpha=alpha,
            eta=eta,
            random_state=self.settings.lda_random_state,
            chunksize=self.settings.lda_chunksize,
            workers=workers,
            per_word_topics=True,
        )
        
        training_time = time.time() - start_time
        
        # Calculate coherence
        coherence = self.calculate_coherence(tokenized_docs)
        
        # Store metadata
        self.metadata = ModelMetadata(
            num_topics=num_topics,
            num_documents=len(tokenized_docs),
            vocabulary_size=len(self.dictionary),
            passes=passes,
            iterations=iterations,
            alpha=alpha,
            eta=eta,
            coherence_score=coherence,
            training_time_seconds=training_time,
        )
        
        # Extract topics
        self._extract_topics()
        
        logger.info(f"Training complete in {training_time:.2f}s")
        logger.info(f"Coherence score: {coherence:.4f}")
        
        return self.model
    
    def calculate_coherence(
        self,
        tokenized_docs: list[list[str]],
        coherence_type: str = "c_v",
    ) -> float:
        """
        Calculate coherence score for the model.
        
        Args:
            tokenized_docs: List of tokenized documents
            coherence_type: Type of coherence measure
            
        Returns:
            Coherence score
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        coherence_model = CoherenceModel(
            model=self.model,
            texts=tokenized_docs,
            dictionary=self.dictionary,
            coherence=coherence_type,
        )
        
        return coherence_model.get_coherence()
    
    def _extract_topics(self, num_words: int = 20) -> None:
        """Extract topic information from the model."""
        if self.model is None:
            return
        
        self._topics = []
        
        for topic_id in range(self.model.num_topics):
            words = self.model.show_topic(topic_id, topn=num_words)
            
            # Generate automatic label from top 3 words
            top_3 = [word for word, _ in words[:3]]
            label = f"Topic {topic_id}: {', '.join(top_3)}"
            
            self._topics.append(TopicInfo(
                topic_id=topic_id,
                words=words,
                label=label,
            ))
    
    def get_topics(self, num_words: int = 10) -> list[TopicInfo]:
        """
        Get all topics with their top words.
        
        Args:
            num_words: Number of top words per topic
            
        Returns:
            List of TopicInfo objects
        """
        if not self._topics:
            self._extract_topics(num_words)
        
        return self._topics
    
    def get_topic_words(self, topic_id: int, num_words: int = 10) -> list[tuple[str, float]]:
        """
        Get top words for a specific topic.
        
        Args:
            topic_id: Topic ID
            num_words: Number of words to return
            
        Returns:
            List of (word, weight) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.show_topic(topic_id, topn=num_words)
    
    def get_document_topics(
        self,
        tokenized_doc: list[str],
        minimum_probability: float = 0.0,
    ) -> list[tuple[int, float]]:
        """
        Get topic distribution for a document.
        
        Args:
            tokenized_doc: Tokenized document
            minimum_probability: Minimum probability threshold
            
        Returns:
            List of (topic_id, probability) tuples
        """
        if self.model is None or self.dictionary is None:
            raise ValueError("Model not trained yet")
        
        bow = self.dictionary.doc2bow(tokenized_doc)
        return self.model.get_document_topics(
            bow,
            minimum_probability=minimum_probability,
        )
    
    def get_document_topic_matrix(
        self,
        tokenized_docs: list[list[str]],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Get topic distribution matrix for all documents.
        
        Args:
            tokenized_docs: List of tokenized documents
            show_progress: Show progress bar
            
        Returns:
            NumPy array of shape (num_docs, num_topics)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        num_docs = len(tokenized_docs)
        num_topics = self.model.num_topics
        
        matrix = np.zeros((num_docs, num_topics))
        
        doc_iter = tqdm(
            enumerate(tokenized_docs),
            total=num_docs,
            desc="Computing topic distributions",
            disable=not show_progress,
        )
        
        for i, doc in doc_iter:
            topics = self.get_document_topics(doc)
            for topic_id, prob in topics:
                matrix[i, topic_id] = prob
        
        return matrix
    
    def infer_topics(
        self,
        text: str,
        preprocessor: Optional[Any] = None,
        num_words: int = 5,
    ) -> dict[str, Any]:
        """
        Infer topics for new text.
        
        Args:
            text: Raw text to analyze
            preprocessor: Preprocessor instance for tokenization
            num_words: Number of top words to include
            
        Returns:
            Dictionary with topic distribution and details
        """
        if self.model is None or self.dictionary is None:
            raise ValueError("Model not trained yet")
        
        # Preprocess if preprocessor provided
        if preprocessor:
            tokens = preprocessor.preprocess_text(text)
        else:
            # Basic tokenization
            tokens = text.lower().split()
        
        # Get topic distribution
        topics = self.get_document_topics(tokens)
        
        # Build result
        result = {
            "tokens": tokens,
            "num_tokens": len(tokens),
            "topics": [],
        }
        
        for topic_id, prob in sorted(topics, key=lambda x: x[1], reverse=True):
            words = self.get_topic_words(topic_id, num_words)
            result["topics"].append({
                "topic_id": topic_id,
                "probability": float(prob),
                "top_words": [w for w, _ in words],
            })
        
        return result
    
    def save(
        self,
        model_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
    ) -> Path:
        """
        Save model, dictionary, corpus, and metadata.
        
        Args:
            model_dir: Directory to save model (uses default if not provided)
            model_name: Base name for model files
            
        Returns:
            Path to saved model directory
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        model_dir = model_dir or self.settings.models_dir
        model_name = model_name or self.settings.lda_model_file
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / f"{model_name}.model"
        self.model.save(str(model_path))
        
        # Save dictionary
        dict_path = model_dir / self.settings.dictionary_file
        self.dictionary.save(str(dict_path))
        
        # Save corpus
        corpus_path = model_dir / self.settings.corpus_file
        corpora.MmCorpus.serialize(str(corpus_path), self.corpus)
        
        # Save metadata
        if self.metadata:
            metadata_path = model_dir / f"{model_name}_metadata.json"
            self.metadata.save(metadata_path)
        
        # Save topics
        topics_path = model_dir / f"{model_name}_topics.json"
        with open(topics_path, "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in self._topics], f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
        
        return model_dir
    
    def load(
        self,
        model_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Load model, dictionary, and metadata.
        
        Args:
            model_dir: Directory containing model files
            model_name: Base name for model files
        """
        model_dir = model_dir or self.settings.models_dir
        model_name = model_name or self.settings.lda_model_file
        
        # Load model
        model_path = model_dir / f"{model_name}.model"
        self.model = LdaMulticore.load(str(model_path))
        
        # Load dictionary
        dict_path = model_dir / self.settings.dictionary_file
        self.dictionary = corpora.Dictionary.load(str(dict_path))
        
        # Load corpus if exists
        corpus_path = model_dir / self.settings.corpus_file
        if corpus_path.exists():
            self.corpus = list(corpora.MmCorpus(str(corpus_path)))
        
        # Load metadata if exists
        metadata_path = model_dir / f"{model_name}_metadata.json"
        if metadata_path.exists():
            self.metadata = ModelMetadata.load(metadata_path)
        
        # Extract topics
        self._extract_topics()
        
        logger.info(f"Model loaded from {model_dir}")


def find_optimal_topics(
    tokenized_docs: list[list[str]],
    topic_range: range = range(5, 21),
    settings: Optional[Settings] = None,
    show_progress: bool = True,
) -> tuple[int, list[dict[str, Any]]]:
    """
    Find optimal number of topics using coherence score.
    
    Args:
        tokenized_docs: List of tokenized documents
        topic_range: Range of topic numbers to try
        settings: Project settings
        show_progress: Show progress bar
        
    Returns:
        Tuple of (optimal_num_topics, results_list)
    """
    settings = settings or get_settings()
    results = []
    
    topic_iter = tqdm(topic_range, desc="Testing topic counts", disable=not show_progress)
    
    for num_topics in topic_iter:
        model = LDATopicModel(settings)
        model.train(
            tokenized_docs,
            num_topics=num_topics,
            show_progress=False,
        )
        
        coherence = model.metadata.coherence_score
        
        results.append({
            "num_topics": num_topics,
            "coherence": coherence,
        })
        
        topic_iter.set_postfix({"topics": num_topics, "coherence": f"{coherence:.4f}"})
    
    # Find optimal
    best = max(results, key=lambda x: x["coherence"])
    optimal = best["num_topics"]
    
    logger.info(f"Optimal number of topics: {optimal} (coherence: {best['coherence']:.4f})")
    
    return optimal, results
