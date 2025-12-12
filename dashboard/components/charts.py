"""
Chart components for the dashboard.

Provides Plotly chart generation functions.
"""

from typing import Optional
from io import BytesIO
import base64

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud


def create_wordcloud_figure(
    word_weights: list[tuple[str, float]],
    width: int = 800,
    height: int = 400,
    colormap: str = "viridis",
) -> str:
    """
    Create a word cloud and return as base64 image.
    
    Args:
        word_weights: List of (word, weight) tuples
        width: Image width
        height: Image height
        colormap: Matplotlib colormap
        
    Returns:
        Base64 encoded PNG image
    """
    word_freq = {word: weight for word, weight in word_weights}
    
    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        colormap=colormap,
        max_words=len(word_freq),
        prefer_horizontal=0.7,
    )
    wc.generate_from_frequencies(word_freq)
    
    # Convert to base64
    buffer = BytesIO()
    wc.to_image().save(buffer, format="PNG")
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode()


def create_topic_bar_chart(
    words: list[str],
    weights: list[float],
    topic_id: int,
    color: str = "#1f77b4",
    height: int = 400,
) -> go.Figure:
    """
    Create a horizontal bar chart of topic words.
    
    Args:
        words: List of words
        weights: List of weights
        topic_id: Topic ID for title
        color: Bar color
        height: Chart height
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=weights,
        y=words,
        orientation='h',
        marker_color=color,
        hovertemplate="<b>%{y}</b><br>Weight: %{x:.4f}<extra></extra>",
    ))
    
    fig.update_layout(
        title=f"Topic {topic_id} - Top Words",
        xaxis_title="Weight",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        height=height,
        margin=dict(l=10, r=10, t=40, b=40),
        showlegend=False,
    )
    
    return fig


def create_topic_prevalence_chart(
    prevalence_df: pd.DataFrame,
    height: int = 400,
) -> go.Figure:
    """
    Create a bar chart of topic prevalence.
    
    Args:
        prevalence_df: DataFrame with topic_id and mean_probability
        height: Chart height
        
    Returns:
        Plotly Figure
    """
    fig = px.bar(
        prevalence_df,
        x="topic_id",
        y="mean_probability",
        color="topic_id",
        error_y="std_probability" if "std_probability" in prevalence_df.columns else None,
        title="Topic Prevalence Across Documents",
        labels={
            "topic_id": "Topic",
            "mean_probability": "Mean Probability",
        },
        color_continuous_scale="Viridis",
        height=height,
    )
    
    fig.update_layout(
        xaxis=dict(tickmode="linear"),
        showlegend=False,
        coloraxis_showscale=False,
    )
    
    return fig


def create_topic_trend_chart(
    trends_df: pd.DataFrame,
    selected_topics: Optional[list[int]] = None,
    date_column: str = "_date",
    height: int = 500,
) -> go.Figure:
    """
    Create a line chart of topic trends over time.
    
    Args:
        trends_df: DataFrame with date and topic columns
        selected_topics: List of topic IDs to show (None for all)
        date_column: Name of date column
        height: Chart height
        
    Returns:
        Plotly Figure
    """
    # Get topic columns
    topic_columns = [col for col in trends_df.columns if col.startswith("topic_")]
    
    if selected_topics is not None:
        topic_columns = [f"topic_{t}" for t in selected_topics if f"topic_{t}" in trends_df.columns]
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, col in enumerate(topic_columns):
        topic_id = col.split("_")[1]
        
        fig.add_trace(go.Scatter(
            x=trends_df[date_column],
            y=trends_df[col],
            name=f"Topic {topic_id}",
            mode="lines+markers",
            line=dict(color=colors[i % len(colors)]),
            hovertemplate=f"<b>Topic {topic_id}</b><br>" +
                         "Date: %{x}<br>" +
                         "Prevalence: %{y:.3f}<extra></extra>",
        ))
    
    fig.update_layout(
        title="Topic Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Topic Prevalence",
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    
    return fig


def create_similarity_chart(
    similarities: list[float],
    labels: list[str],
    title: str = "Document Similarity",
    height: int = 400,
) -> go.Figure:
    """
    Create a horizontal bar chart of similarity scores.
    
    Args:
        similarities: List of similarity scores
        labels: List of document labels
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly Figure
    """
    # Color based on similarity
    colors = ["#27ae60" if s > 0.7 else "#f39c12" if s > 0.4 else "#e74c3c" for s in similarities]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=similarities,
        y=labels,
        orientation='h',
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Similarity: %{x:.3f}<extra></extra>",
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Similarity Score",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        height=height,
        xaxis=dict(range=[0, 1]),
        margin=dict(l=10, r=10, t=40, b=40),
    )
    
    return fig


def create_topic_distribution_pie(
    topic_probs: list[tuple[int, float]],
    height: int = 400,
) -> go.Figure:
    """
    Create a pie chart of topic distribution for a document.
    
    Args:
        topic_probs: List of (topic_id, probability) tuples
        height: Chart height
        
    Returns:
        Plotly Figure
    """
    # Filter out very small probabilities
    topic_probs = [(t, p) for t, p in topic_probs if p > 0.01]
    
    topics = [f"Topic {t}" for t, _ in topic_probs]
    probs = [p for _, p in topic_probs]
    
    fig = go.Figure(data=[go.Pie(
        labels=topics,
        values=probs,
        hovertemplate="<b>%{label}</b><br>Probability: %{value:.3f}<extra></extra>",
        textinfo="label+percent",
        hole=0.3,
    )])
    
    fig.update_layout(
        title="Topic Distribution",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    return fig


def create_topic_comparison_chart(
    topic_a_words: list[tuple[str, float]],
    topic_b_words: list[tuple[str, float]],
    topic_a_id: int,
    topic_b_id: int,
    height: int = 500,
) -> go.Figure:
    """
    Create a side-by-side comparison of two topics.
    
    Args:
        topic_a_words: Words and weights for topic A
        topic_b_words: Words and weights for topic B
        topic_a_id: Topic A ID
        topic_b_id: Topic B ID
        height: Chart height
        
    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Topic {topic_a_id}", f"Topic {topic_b_id}"],
        horizontal_spacing=0.15,
    )
    
    # Topic A
    words_a = [w for w, _ in topic_a_words[:10]]
    weights_a = [w for _, w in topic_a_words[:10]]
    
    fig.add_trace(
        go.Bar(
            x=weights_a,
            y=words_a,
            orientation='h',
            marker_color="#1f77b4",
            name=f"Topic {topic_a_id}",
            showlegend=False,
        ),
        row=1, col=1
    )
    
    # Topic B
    words_b = [w for w, _ in topic_b_words[:10]]
    weights_b = [w for _, w in topic_b_words[:10]]
    
    fig.add_trace(
        go.Bar(
            x=weights_b,
            y=words_b,
            orientation='h',
            marker_color="#ff7f0e",
            name=f"Topic {topic_b_id}",
            showlegend=False,
        ),
        row=1, col=2
    )
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=height,
        title="Topic Comparison",
        margin=dict(l=10, r=10, t=60, b=40),
    )
    
    return fig


def create_heatmap(
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str = "Heatmap",
    colorscale: str = "Viridis",
    height: int = 500,
) -> go.Figure:
    """
    Create a heatmap visualization.
    
    Args:
        matrix: 2D numpy array
        x_labels: X-axis labels
        y_labels: Y-axis labels
        title: Chart title
        colorscale: Plotly colorscale
        height: Chart height
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        hovertemplate="%{x}<br>%{y}<br>Value: %{z:.3f}<extra></extra>",
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40, b=40),
    )
    
    return fig
