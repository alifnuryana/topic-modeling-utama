"""
Komponen filter untuk dashboard.

Menyediakan widget filter yang dapat digunakan kembali untuk Streamlit.
"""

from typing import Optional, Tuple
from datetime import date

import streamlit as st


def topic_selector(
    num_topics: int,
    key: str = "topic_select",
    label: str = "Pilih Topik",
    default: int = 0,
    include_all: bool = False,
    labels: Optional[dict[int, str]] = None,
) -> Optional[int]:
    """
    Membuat widget pemilih topik.
    
    Args:
        num_topics: Total jumlah topik
        key: Key widget
        label: Label widget
        default: Pilihan topik default
        include_all: Sertakan opsi "Semua Topik"
        labels: Dict mapping topic_id -> label kustom (opsional)
        
    Returns:
        ID topik terpilih atau None jika "Semua" dipilih
    """
    options = list(range(num_topics))
    
    if include_all:
        options = ["Semua Topik"] + options
    
    def format_topic(x):
        if isinstance(x, int):
            if labels and x in labels:
                return f"Topik {x}: {labels[x]}"
            return f"Topik {x}"
        return x
    
    selection = st.selectbox(
        label,
        options=options,
        index=default if not include_all else default + 1,
        key=key,
        format_func=format_topic,
    )
    
    return None if selection == "Semua Topik" else selection


def multi_topic_selector(
    num_topics: int,
    key: str = "multi_topic_select",
    label: str = "Pilih Topik",
    default: Optional[list[int]] = None,
    labels: Optional[dict[int, str]] = None,
) -> list[int]:
    """
    Membuat widget pemilih multi-topik.
    
    Args:
        num_topics: Total jumlah topik
        key: Key widget
        label: Label widget
        default: Pilihan default
        labels: Dict mapping topic_id -> label kustom (opsional)
        
    Returns:
        Daftar ID topik terpilih
    """
    options = list(range(num_topics))
    default = default or options[:3]  # Default to first 3
    
    def format_topic(x):
        if labels and x in labels:
            return f"Topik {x}: {labels[x]}"
        return f"Topik {x}"
    
    selection = st.multiselect(
        label,
        options=options,
        default=default,
        key=key,
        format_func=format_topic,
    )
    
    return selection


def date_range_selector(
    min_year: int,
    max_year: int,
    key: str = "date_range",
    label: str = "Pilih Rentang Tahun",
) -> Tuple[int, int]:
    """
    Membuat slider rentang tahun.
    
    Args:
        min_year: Tahun minimum
        max_year: Tahun maksimum
        key: Key widget
        label: Label widget
        
    Returns:
        Tuple (tahun_awal, tahun_akhir)
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
    label: str = "Cari",
    placeholder: str = "Masukkan kata pencarian...",
) -> str:
    """
    Membuat widget input pencarian.
    
    Args:
        key: Key widget
        label: Label widget
        placeholder: Placeholder input
        
    Returns:
        String query pencarian
    """
    return st.text_input(
        label,
        placeholder=placeholder,
        key=key,
    )


def probability_slider(
    key: str = "prob_threshold",
    label: str = "Probabilitas Topik Minimum",
    default: float = 0.1,
    min_value: float = 0.0,
    max_value: float = 1.0,
    step: float = 0.05,
) -> float:
    """
    Membuat slider threshold probabilitas.
    
    Args:
        key: Key widget
        label: Label widget
        default: Nilai default
        min_value: Nilai minimum
        max_value: Nilai maksimum
        step: Ukuran langkah
        
    Returns:
        Threshold probabilitas terpilih
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
    label: str = "Jumlah Hasil",
    options: list[int] = [5, 10, 20, 50, 100],
    default_index: int = 1,
) -> int:
    """
    Membuat pemilih jumlah hasil.
    
    Args:
        key: Key widget
        label: Label widget
        options: Daftar opsi
        default_index: Indeks pilihan default
        
    Returns:
        Jumlah terpilih
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
    label: str = "Urutkan Berdasarkan",
    default_index: int = 0,
) -> str:
    """
    Membuat pemilih pengurutan.
    
    Args:
        options: Dict dari nama_tampilan -> nama_kolom
        key: Key widget
        label: Label widget
        default_index: Indeks pilihan default
        
    Returns:
        Nama kolom terpilih
    """
    display_names = list(options.keys())
    
    selection = st.selectbox(
        label,
        options=display_names,
        index=default_index,
        key=key,
    )
    
    return options[selection]
