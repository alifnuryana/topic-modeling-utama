# Placeholder untuk Gambar 4.1

## Informasi Gambar
- **Judul**: Diagram Alur Proses Pengumpulan Data
- **Deskripsi**: Flowchart yang menunjukkan proses harvesting data dari repositori OAI-PMH

## Diagram Konseptual

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION FLOW                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  OAI-PMH    │────▶│  cloudscraper   │────▶│    Sickle       │
│  Endpoint   │     │  (Anti-bot)     │     │    (Parser)     │
└─────────────┘     └─────────────────┘     └─────────────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────┐
│                        OAIPMHHarvester                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  identify() │  │ list_sets() │  │ harvest_and_save()  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     Metadata Extraction                      │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌──────┐  │
│  │ Title   │ │Abstract │ │ Authors  │ │ Date    │ │ etc. │  │
│  └─────────┘ └─────────┘ └──────────┘ └─────────┘ └──────┘  │
└─────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────┐
│               data/raw/raw_metadata.csv                      │
│               (12.647 records, 24.9 MB)                      │
└─────────────────────────────────────────────────────────────┘
```

## Langkah-langkah Proses:
1. **Konfigurasi Endpoint** - URL repositori OAI-PMH
2. **Anti-bot Bypass** - cloudscraper menangani proteksi
3. **Identify Repository** - Validasi koneksi dan info dasar
4. **List Sets** - Identifikasi koleksi yang tersedia
5. **Harvest Records** - Iterasi mengambil metadata
6. **Parse XML** - Ekstrak field Dublin Core
7. **Save to CSV** - Simpan dalam format tabular
