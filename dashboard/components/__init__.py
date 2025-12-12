"""
Dashboard components package.

Reusable UI components for the Streamlit dashboard.
"""

from dashboard.components.charts import (
    create_wordcloud_figure,
    create_topic_bar_chart,
    create_topic_prevalence_chart,
    create_topic_trend_chart,
    create_similarity_chart,
)

from dashboard.components.filters import (
    topic_selector,
    date_range_selector,
    search_input,
    probability_slider,
)

__all__ = [
    "create_wordcloud_figure",
    "create_topic_bar_chart",
    "create_topic_prevalence_chart",
    "create_topic_trend_chart",
    "create_similarity_chart",
    "topic_selector",
    "date_range_selector",
    "search_input",
    "probability_slider",
]
