"""
Filter components for the dashboard.

Provides reusable filter widgets for Streamlit.
"""

from typing import Optional, Tuple
from datetime import date

import streamlit as st


def topic_selector(
    num_topics: int,
    key: str = "topic_select",
    label: str = "Select Topic",
    default: int = 0,
    include_all: bool = False,
) -> Optional[int]:
    """
    Create a topic selector widget.
    
    Args:
        num_topics: Total number of topics
        key: Widget key
        label: Widget label
        default: Default topic selection
        include_all: Include "All Topics" option
        
    Returns:
        Selected topic ID or None if "All" selected
    """
    options = list(range(num_topics))
    
    if include_all:
        options = ["All Topics"] + options
        
    selection = st.selectbox(
        label,
        options=options,
        index=default if not include_all else default + 1,
        key=key,
        format_func=lambda x: f"Topic {x}" if isinstance(x, int) else x,
    )
    
    return None if selection == "All Topics" else selection


def multi_topic_selector(
    num_topics: int,
    key: str = "multi_topic_select",
    label: str = "Select Topics",
    default: Optional[list[int]] = None,
) -> list[int]:
    """
    Create a multi-topic selector widget.
    
    Args:
        num_topics: Total number of topics
        key: Widget key
        label: Widget label
        default: Default selections
        
    Returns:
        List of selected topic IDs
    """
    options = list(range(num_topics))
    default = default or options[:3]  # Default to first 3
    
    selection = st.multiselect(
        label,
        options=options,
        default=default,
        key=key,
        format_func=lambda x: f"Topic {x}",
    )
    
    return selection


def date_range_selector(
    min_year: int,
    max_year: int,
    key: str = "date_range",
    label: str = "Select Year Range",
) -> Tuple[int, int]:
    """
    Create a year range slider.
    
    Args:
        min_year: Minimum year
        max_year: Maximum year
        key: Widget key
        label: Widget label
        
    Returns:
        Tuple of (start_year, end_year)
    """
    return st.slider(
        label,
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        key=key,
    )


def search_input(
    key: str = "search",
    label: str = "Search",
    placeholder: str = "Enter search terms...",
) -> str:
    """
    Create a search input widget.
    
    Args:
        key: Widget key
        label: Widget label
        placeholder: Input placeholder
        
    Returns:
        Search query string
    """
    return st.text_input(
        label,
        placeholder=placeholder,
        key=key,
    )


def probability_slider(
    key: str = "prob_threshold",
    label: str = "Minimum Topic Probability",
    default: float = 0.1,
    min_value: float = 0.0,
    max_value: float = 1.0,
    step: float = 0.05,
) -> float:
    """
    Create a probability threshold slider.
    
    Args:
        key: Widget key
        label: Widget label
        default: Default value
        min_value: Minimum value
        max_value: Maximum value
        step: Step size
        
    Returns:
        Selected probability threshold
    """
    return st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default,
        step=step,
        key=key,
    )


def num_results_selector(
    key: str = "num_results",
    label: str = "Number of Results",
    options: list[int] = [5, 10, 20, 50, 100],
    default_index: int = 1,
) -> int:
    """
    Create a number of results selector.
    
    Args:
        key: Widget key
        label: Widget label
        options: List of options
        default_index: Default selection index
        
    Returns:
        Selected number
    """
    return st.selectbox(
        label,
        options=options,
        index=default_index,
        key=key,
    )


def sort_selector(
    options: dict[str, str],
    key: str = "sort_by",
    label: str = "Sort By",
    default_index: int = 0,
) -> str:
    """
    Create a sort by selector.
    
    Args:
        options: Dict of display_name -> column_name
        key: Widget key
        label: Widget label
        default_index: Default selection index
        
    Returns:
        Selected column name
    """
    display_names = list(options.keys())
    
    selection = st.selectbox(
        label,
        options=display_names,
        index=default_index,
        key=key,
    )
    
    return options[selection]
