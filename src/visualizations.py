"""
Visualization module for Topic Modeling.

Provides functions for creating visualizations including word clouds,
topic charts, pyLDAvis, and trend analysis plots.
"""

import logging
from pathlib import Path
from typing import Optional, Any
from io import BytesIO
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import Settings, get_settings
from src.lda_model import LDATopicModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8-whitegrid")


class TopicVisualizer:
    """
    Visualization generator for topic models.
    
    Provides methods for creating:
    - Word clouds
    - Topic distribution charts
    - pyLDAvis interactive visualization
    - Topic trend charts
    - Document-topic heatmaps
    """
    
    # Color palette for topics
    TOPIC_COLORS = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    
    def __init__(
        self,
        model: Optional[LDATopicModel] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize the visualizer.
        
        Args:
            model: Trained LDA topic model
            settings: Project settings
        """
        self.model = model
        self.settings = settings or get_settings()
    
    def set_model(self, model: LDATopicModel) -> None:
        """Set the topic model."""
        self.model = model
    
    def _get_topic_color(self, topic_id: int) -> str:
        """Get consistent color for a topic."""
        return self.TOPIC_COLORS[topic_id % len(self.TOPIC_COLORS)]
    
    def create_wordcloud(
        self,
        topic_id: int,
        num_words: int = 50,
        width: int = 800,
        height: int = 400,
        background_color: str = "white",
        colormap: str = "viridis",
        save_path: Optional[Path] = None,
    ) -> WordCloud:
        """
        Create a word cloud for a topic.
        
        Args:
            topic_id: Topic ID
            num_words: Number of words to include
            width: Image width
            height: Image height
            background_color: Background color
            colormap: Matplotlib colormap name
            save_path: Optional path to save image
            
        Returns:
            WordCloud object
        """
        if self.model is None or self.model.model is None:
            raise ValueError("Model not set")
        
        # Get word weights for topic
        words = self.model.get_topic_words(topic_id, num_words)
        word_freq = {word: weight for word, weight in words}
        
        # Create word cloud
        wc = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            colormap=colormap,
            max_words=num_words,
            prefer_horizontal=0.7,
        )
        wc.generate_from_frequencies(word_freq)
        
        if save_path:
            wc.to_file(str(save_path))
            logger.info(f"Saved word cloud to {save_path}")
        
        return wc
    
    def plot_wordcloud(
        self,
        topic_id: int,
        num_words: int = 50,
        figsize: tuple[int, int] = (12, 6),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot a word cloud for a topic.
        
        Args:
            topic_id: Topic ID
            num_words: Number of words
            figsize: Figure size
            title: Plot title
            
        Returns:
            Matplotlib Figure
        """
        wc = self.create_wordcloud(topic_id, num_words)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        
        if title is None:
            title = f"Topic {topic_id} Word Cloud"
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        return fig
    
    def plot_all_wordclouds(
        self,
        num_words: int = 30,
        cols: int = 3,
        figsize_per_plot: tuple[int, int] = (5, 3),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot word clouds for all topics in a grid.
        
        Args:
            num_words: Number of words per cloud
            cols: Number of columns in grid
            figsize_per_plot: Size per subplot
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib Figure
        """
        if self.model is None or self.model.model is None:
            raise ValueError("Model not set")
        
        num_topics = self.model.model.num_topics
        rows = (num_topics + cols - 1) // cols
        
        figsize = (figsize_per_plot[0] * cols, figsize_per_plot[1] * rows)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_topics > 1 else [axes]
        
        for i in range(num_topics):
            wc = self.create_wordcloud(i, num_words)
            axes[i].imshow(wc, interpolation="bilinear")
            axes[i].axis("off")
            axes[i].set_title(f"Topic {i}", fontsize=10)
        
        # Hide empty subplots
        for i in range(num_topics, len(axes)):
            axes[i].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved word clouds to {save_path}")
        
        return fig
    
    def plot_topic_words_bar(
        self,
        topic_id: int,
        num_words: int = 15,
        figsize: tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Plot horizontal bar chart of topic words.
        
        Args:
            topic_id: Topic ID
            num_words: Number of words
            figsize: Figure size
            
        Returns:
            Matplotlib Figure
        """
        if self.model is None:
            raise ValueError("Model not set")
        
        words = self.model.get_topic_words(topic_id, num_words)
        words_list = [w for w, _ in words]
        weights = [weight for _, weight in words]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words_list)))
        
        y_pos = np.arange(len(words_list))
        ax.barh(y_pos, weights, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words_list)
        ax.invert_yaxis()
        ax.set_xlabel("Weight")
        ax.set_title(f"Topic {topic_id} - Top {num_words} Words", fontweight="bold")
        
        plt.tight_layout()
        return fig
    
    def plot_topic_prevalence(
        self,
        topic_prevalence: pd.DataFrame,
        figsize: tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot topic prevalence bar chart.
        
        Args:
            topic_prevalence: DataFrame with topic prevalence data
            figsize: Figure size
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        topics = topic_prevalence["topic_id"].values
        prevalence = topic_prevalence["mean_probability"].values
        
        colors = [self._get_topic_color(t) for t in topics]
        
        ax.bar(topics, prevalence, color=colors)
        ax.set_xlabel("Topic ID")
        ax.set_ylabel("Mean Probability")
        ax.set_title("Topic Prevalence Across Documents", fontweight="bold")
        ax.set_xticks(topics)
        
        plt.tight_layout()
        return fig
    
    def plot_topic_prevalence_plotly(
        self,
        topic_prevalence: pd.DataFrame,
    ) -> go.Figure:
        """
        Create interactive topic prevalence chart with Plotly.
        
        Args:
            topic_prevalence: DataFrame with topic prevalence data
            
        Returns:
            Plotly Figure
        """
        fig = px.bar(
            topic_prevalence,
            x="topic_id",
            y="mean_probability",
            error_y="std_probability",
            color="topic_id",
            title="Topic Prevalence Across Documents",
            labels={
                "topic_id": "Topic ID",
                "mean_probability": "Mean Probability",
            },
        )
        
        fig.update_layout(
            xaxis=dict(tickmode="linear"),
            showlegend=False,
        )
        
        return fig
    
    def plot_topic_trends(
        self,
        trends_df: pd.DataFrame,
        date_column: str = "_date",
        figsize: tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """
        Plot topic trends over time.
        
        Args:
            trends_df: DataFrame with topic trends data
            date_column: Date column name
            figsize: Figure size
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get topic columns
        topic_columns = [col for col in trends_df.columns if col.startswith("topic_")]
        
        for col in topic_columns:
            topic_id = int(col.split("_")[1])
            ax.plot(
                trends_df[date_column],
                trends_df[col],
                label=f"Topic {topic_id}",
                color=self._get_topic_color(topic_id),
                marker="o",
                markersize=4,
            )
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Topic Prevalence")
        ax.set_title("Topic Trends Over Time", fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_topic_trends_plotly(
        self,
        trends_df: pd.DataFrame,
        date_column: str = "_date",
    ) -> go.Figure:
        """
        Create interactive topic trends chart with Plotly.
        
        Args:
            trends_df: DataFrame with topic trends data
            date_column: Date column name
            
        Returns:
            Plotly Figure
        """
        # Get topic columns
        topic_columns = [col for col in trends_df.columns if col.startswith("topic_")]
        
        fig = go.Figure()
        
        for col in topic_columns:
            topic_id = int(col.split("_")[1])
            fig.add_trace(go.Scatter(
                x=trends_df[date_column],
                y=trends_df[col],
                name=f"Topic {topic_id}",
                mode="lines+markers",
            ))
        
        fig.update_layout(
            title="Topic Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Topic Prevalence",
            hovermode="x unified",
        )
        
        return fig
    
    def plot_document_topic_heatmap(
        self,
        topic_doc_matrix: np.ndarray,
        num_docs: int = 50,
        figsize: tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """
        Plot heatmap of document-topic distributions.
        
        Args:
            topic_doc_matrix: Document-topic matrix
            num_docs: Number of documents to show
            figsize: Figure size
            
        Returns:
            Matplotlib Figure
        """
        # Take first num_docs documents
        matrix = topic_doc_matrix[:num_docs]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            matrix,
            cmap="YlOrRd",
            xticklabels=[f"T{i}" for i in range(matrix.shape[1])],
            yticklabels=False,
            ax=ax,
            cbar_kws={"label": "Probability"},
        )
        
        ax.set_xlabel("Topic")
        ax.set_ylabel("Document")
        ax.set_title(f"Document-Topic Distribution (First {num_docs} Documents)", fontweight="bold")
        
        plt.tight_layout()
        return fig
    
    def plot_topic_distance_heatmap(
        self,
        distance_matrix: np.ndarray,
        figsize: tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Plot heatmap of topic distances.
        
        Args:
            distance_matrix: Topic distance matrix
            figsize: Figure size
            
        Returns:
            Matplotlib Figure
        """
        num_topics = distance_matrix.shape[0]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            distance_matrix,
            cmap="coolwarm_r",
            xticklabels=[f"T{i}" for i in range(num_topics)],
            yticklabels=[f"T{i}" for i in range(num_topics)],
            annot=True,
            fmt=".2f",
            ax=ax,
            cbar_kws={"label": "Jensen-Shannon Distance"},
        )
        
        ax.set_title("Topic Distance Matrix", fontweight="bold")
        
        plt.tight_layout()
        return fig
    
    def create_pyldavis(
        self,
        tokenized_docs: list[list[str]],
        save_path: Optional[Path] = None,
    ) -> Any:
        """
        Create pyLDAvis visualization.
        
        Args:
            tokenized_docs: Tokenized documents
            save_path: Optional path to save HTML
            
        Returns:
            pyLDAvis prepared data
        """
        import pyLDAvis
        import pyLDAvis.gensim_models
        
        if self.model is None or self.model.model is None:
            raise ValueError("Model not set")
        
        # Prepare visualization
        vis_data = pyLDAvis.gensim_models.prepare(
            self.model.model,
            self.model.corpus,
            self.model.dictionary,
            mds="mmds",
        )
        
        if save_path:
            pyLDAvis.save_html(vis_data, str(save_path))
            logger.info(f"Saved pyLDAvis to {save_path}")
        
        return vis_data
    
    def plot_topic_comparison(
        self,
        topic_a: int,
        topic_b: int,
        num_words: int = 15,
        figsize: tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Plot side-by-side comparison of two topics.
        
        Args:
            topic_a: First topic ID
            topic_b: Second topic ID
            num_words: Number of words per topic
            figsize: Figure size
            
        Returns:
            Matplotlib Figure
        """
        if self.model is None:
            raise ValueError("Model not set")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for idx, topic_id in enumerate([topic_a, topic_b]):
            words = self.model.get_topic_words(topic_id, num_words)
            words_list = [w for w, _ in words]
            weights = [weight for _, weight in words]
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words_list)))
            
            y_pos = np.arange(len(words_list))
            axes[idx].barh(y_pos, weights, color=colors)
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(words_list)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel("Weight")
            axes[idx].set_title(f"Topic {topic_id}", fontweight="bold")
        
        plt.suptitle(f"Topic Comparison: {topic_a} vs {topic_b}", fontweight="bold", y=1.02)
        plt.tight_layout()
        
        return fig
    
    def wordcloud_to_base64(
        self,
        topic_id: int,
        num_words: int = 50,
    ) -> str:
        """
        Generate word cloud as base64 string for web embedding.
        
        Args:
            topic_id: Topic ID
            num_words: Number of words
            
        Returns:
            Base64 encoded PNG image
        """
        wc = self.create_wordcloud(topic_id, num_words)
        
        buffer = BytesIO()
        wc.to_image().save(buffer, format="PNG")
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    def save_all_visualizations(
        self,
        tokenized_docs: list[list[str]],
        topic_doc_matrix: np.ndarray,
        topic_prevalence: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> dict[str, Path]:
        """
        Save all visualizations to files.
        
        Args:
            tokenized_docs: Tokenized documents
            topic_doc_matrix: Document-topic matrix
            topic_prevalence: Topic prevalence data
            output_dir: Output directory
            
        Returns:
            Dictionary of visualization paths
        """
        output_dir = output_dir or self.settings.outputs_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Word clouds for all topics
        wc_path = output_dir / "wordclouds_all.png"
        fig = self.plot_all_wordclouds(save_path=wc_path)
        plt.close(fig)
        paths["wordclouds"] = wc_path
        
        # Topic prevalence
        prev_path = output_dir / "topic_prevalence.png"
        fig = self.plot_topic_prevalence(topic_prevalence)
        fig.savefig(prev_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["prevalence"] = prev_path
        
        # Document-topic heatmap
        hm_path = output_dir / "doc_topic_heatmap.png"
        fig = self.plot_document_topic_heatmap(topic_doc_matrix)
        fig.savefig(hm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["heatmap"] = hm_path
        
        # pyLDAvis
        ldavis_path = output_dir / "pyldavis.html"
        self.create_pyldavis(tokenized_docs, save_path=ldavis_path)
        paths["pyldavis"] = ldavis_path
        
        logger.info(f"Saved all visualizations to {output_dir}")
        
        return paths
