"""
Topic Analysis module.

Provides analysis utilities for exploring LDA model results including
document similarity, topic trends, and topic coherence metrics.
"""

import logging
from typing import Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, jensenshannon
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Settings, get_settings
from src.lda_model import LDATopicModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicAnalyzer:
    """
    Analyzer for topic model results.
    
    Provides methods for:
    - Document similarity search
    - Topic trends over time
    - Topic comparison and distance
    - Document-topic analysis
    """
    
    def __init__(
        self,
        model: LDATopicModel,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize the analyzer.
        
        Args:
            model: Trained LDA topic model
            settings: Project settings
        """
        self.model = model
        self.settings = settings or get_settings()
        
        self._topic_doc_matrix: Optional[np.ndarray] = None
        self._document_data: Optional[pd.DataFrame] = None
    
    def set_document_data(
        self,
        df: pd.DataFrame,
        tokens_column: str = "tokens",
    ) -> None:
        """
        Set document data for analysis.
        
        Args:
            df: DataFrame with document data
            tokens_column: Column containing tokenized text
        """
        self._document_data = df.copy()
        
        # Compute topic distributions if not already done
        if self._topic_doc_matrix is None:
            tokenized_docs = df[tokens_column].tolist()
            self._topic_doc_matrix = self.model.get_document_topic_matrix(
                tokenized_docs,
                show_progress=True,
            )
    
    def get_topic_distribution(self, doc_index: int) -> dict[int, float]:
        """
        Get topic distribution for a specific document.
        
        Args:
            doc_index: Document index
            
        Returns:
            Dictionary mapping topic_id to probability
        """
        if self._topic_doc_matrix is None:
            raise ValueError("Document data not set. Call set_document_data first.")
        
        distribution = self._topic_doc_matrix[doc_index]
        return {i: float(prob) for i, prob in enumerate(distribution)}
    
    def get_dominant_topic(self, doc_index: int) -> tuple[int, float]:
        """
        Get the dominant topic for a document.
        
        Args:
            doc_index: Document index
            
        Returns:
            Tuple of (topic_id, probability)
        """
        distribution = self.get_topic_distribution(doc_index)
        topic_id = max(distribution, key=distribution.get)
        return topic_id, distribution[topic_id]
    
    def get_documents_by_topic(
        self,
        topic_id: int,
        min_probability: float = 0.3,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get documents dominated by a specific topic.
        
        Args:
            topic_id: Topic ID to filter
            min_probability: Minimum topic probability
            top_n: Maximum number of documents to return
            
        Returns:
            DataFrame of matching documents
        """
        if self._topic_doc_matrix is None or self._document_data is None:
            raise ValueError("Document data not set. Call set_document_data first.")
        
        # Get probabilities for this topic
        probs = self._topic_doc_matrix[:, topic_id]
        
        # Create result with probabilities
        result = self._document_data.copy()
        result[f"topic_{topic_id}_prob"] = probs
        
        # Filter and sort
        result = result[result[f"topic_{topic_id}_prob"] >= min_probability]
        result = result.sort_values(f"topic_{topic_id}_prob", ascending=False)
        
        if top_n:
            result = result.head(top_n)
        
        return result
    
    def find_similar_documents(
        self,
        doc_index: int,
        top_n: int = 10,
        method: str = "cosine",
    ) -> pd.DataFrame:
        """
        Find documents similar to a given document based on topic distribution.
        
        Args:
            doc_index: Index of the query document
            top_n: Number of similar documents to return
            method: Similarity method ('cosine' or 'jensenshannon')
            
        Returns:
            DataFrame of similar documents with similarity scores
        """
        if self._topic_doc_matrix is None or self._document_data is None:
            raise ValueError("Document data not set. Call set_document_data first.")
        
        query_dist = self._topic_doc_matrix[doc_index].reshape(1, -1)
        
        if method == "cosine":
            similarities = cosine_similarity(query_dist, self._topic_doc_matrix)[0]
        elif method == "jensenshannon":
            # Jensen-Shannon distance (convert to similarity)
            similarities = np.array([
                1 - jensenshannon(query_dist[0], doc_dist)
                for doc_dist in self._topic_doc_matrix
            ])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get top N (excluding the query document itself)
        similar_indices = np.argsort(similarities)[::-1]
        similar_indices = [i for i in similar_indices if i != doc_index][:top_n]
        
        result = self._document_data.iloc[similar_indices].copy()
        result["similarity"] = [similarities[i] for i in similar_indices]
        
        return result
    
    def find_similar_by_text(
        self,
        text: str,
        preprocessor: Any,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Find documents similar to a given text.
        
        Args:
            text: Query text
            preprocessor: Preprocessor for tokenization
            top_n: Number of similar documents to return
            
        Returns:
            DataFrame of similar documents with similarity scores
        """
        if self._topic_doc_matrix is None or self._document_data is None:
            raise ValueError("Document data not set. Call set_document_data first.")
        
        # Get topic distribution for query text
        tokens = preprocessor.preprocess_text(text)
        query_topics = self.model.get_document_topics(tokens)
        
        # Convert to array
        query_dist = np.zeros(self.model.model.num_topics)
        for topic_id, prob in query_topics:
            query_dist[topic_id] = prob
        
        query_dist = query_dist.reshape(1, -1)
        
        # Compute similarities
        similarities = cosine_similarity(query_dist, self._topic_doc_matrix)[0]
        
        # Get top N
        similar_indices = np.argsort(similarities)[::-1][:top_n]
        
        result = self._document_data.iloc[similar_indices].copy()
        result["similarity"] = [similarities[i] for i in similar_indices]
        
        return result
    
    def compute_topic_trends(
        self,
        date_column: str = "date",
        freq: str = "Y",
    ) -> pd.DataFrame:
        """
        Compute topic prevalence trends over time.
        
        Args:
            date_column: Column containing dates
            freq: Frequency for aggregation ('Y' for year, 'M' for month, etc.)
            
        Returns:
            DataFrame with topic prevalence by time period
        """
        if self._topic_doc_matrix is None or self._document_data is None:
            raise ValueError("Document data not set. Call set_document_data first.")
        
        # Create DataFrame with dates and topic distributions
        df = self._document_data.copy()
        
        # Parse dates
        df["_date"] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=["_date"])
        
        # Add topic columns
        num_topics = self._topic_doc_matrix.shape[1]
        for i in range(num_topics):
            df[f"topic_{i}"] = self._topic_doc_matrix[df.index, i]
        
        # Set date as index and resample
        df = df.set_index("_date")
        
        topic_columns = [f"topic_{i}" for i in range(num_topics)]
        trends = df[topic_columns].resample(freq).mean()
        
        return trends.reset_index()
    
    def compute_topic_prevalence(self) -> pd.DataFrame:
        """
        Compute overall topic prevalence across all documents.
        
        Returns:
            DataFrame with topic prevalence statistics
        """
        if self._topic_doc_matrix is None:
            raise ValueError("Document data not set. Call set_document_data first.")
        
        num_topics = self._topic_doc_matrix.shape[1]
        
        prevalence = []
        for i in range(num_topics):
            topic_probs = self._topic_doc_matrix[:, i]
            
            prevalence.append({
                "topic_id": i,
                "mean_probability": float(np.mean(topic_probs)),
                "std_probability": float(np.std(topic_probs)),
                "max_probability": float(np.max(topic_probs)),
                "num_dominant": int(np.sum(np.argmax(self._topic_doc_matrix, axis=1) == i)),
            })
        
        return pd.DataFrame(prevalence)
    
    def compute_topic_distance_matrix(self) -> np.ndarray:
        """
        Compute pairwise distances between topics.
        
        Returns:
            Distance matrix of shape (num_topics, num_topics)
        """
        if self.model.model is None:
            raise ValueError("Model not trained")
        
        num_topics = self.model.model.num_topics
        
        # Get topic-word distributions
        topic_word_matrix = self.model.model.get_topics()
        
        # Compute pairwise Jensen-Shannon distances
        distances = np.zeros((num_topics, num_topics))
        
        for i in range(num_topics):
            for j in range(num_topics):
                if i != j:
                    distances[i, j] = jensenshannon(
                        topic_word_matrix[i],
                        topic_word_matrix[j],
                    )
        
        return distances
    
    def get_topic_overlap(
        self,
        topic_a: int,
        topic_b: int,
        num_words: int = 20,
    ) -> dict[str, Any]:
        """
        Analyze overlap between two topics.
        
        Args:
            topic_a: First topic ID
            topic_b: Second topic ID
            num_words: Number of top words to consider
            
        Returns:
            Dictionary with overlap analysis
        """
        words_a = set(word for word, _ in self.model.get_topic_words(topic_a, num_words))
        words_b = set(word for word, _ in self.model.get_topic_words(topic_b, num_words))
        
        intersection = words_a & words_b
        union = words_a | words_b
        
        return {
            "topic_a": topic_a,
            "topic_b": topic_b,
            "words_a": list(words_a),
            "words_b": list(words_b),
            "shared_words": list(intersection),
            "unique_to_a": list(words_a - words_b),
            "unique_to_b": list(words_b - words_a),
            "jaccard_similarity": len(intersection) / len(union) if union else 0,
        }
    
    def create_topic_document_matrix_df(self) -> pd.DataFrame:
        """
        Create a DataFrame with document info and topic probabilities.
        
        Returns:
            DataFrame with document data and topic columns
        """
        if self._topic_doc_matrix is None or self._document_data is None:
            raise ValueError("Document data not set. Call set_document_data first.")
        
        result = self._document_data.copy()
        
        # Add topic columns
        num_topics = self._topic_doc_matrix.shape[1]
        for i in range(num_topics):
            result[f"topic_{i}"] = self._topic_doc_matrix[:, i]
        
        # Add dominant topic
        result["dominant_topic"] = np.argmax(self._topic_doc_matrix, axis=1)
        result["dominant_topic_prob"] = np.max(self._topic_doc_matrix, axis=1)
        
        return result
    
    def save_topic_document_matrix(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Save topic-document matrix to CSV.
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        df = self.create_topic_document_matrix_df()
        
        if output_path is None:
            output_path = str(
                self.settings.processed_data_dir / 
                self.settings.topic_document_matrix_file
            )
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved topic-document matrix to {output_path}")
        
        return output_path
