# BAB 3: METODOLOGI PENELITIAN

## 3.1 Desain Penelitian

Penelitian ini menggunakan pendekatan kuantitatif dengan metode analisis teks otomatis. Metode yang digunakan adalah *topic modeling* dengan algoritma *Latent Dirichlet Allocation* (LDA) untuk mengidentifikasi tema-tema utama dari metadata tugas akhir di repositori Universitas Widyatama.

[Gambar 3.1: Diagram Alur Metodologi Penelitian](placeholder/gambar_3_1_diagram_alur_metodologi.md)

---

## 3.2 Sumber dan Pengumpulan Data

### 3.2.1 Sumber Data

Data penelitian berasal dari repositori institusional Universitas Widyatama melalui protokol **OAI-PMH** (*Open Archives Initiative Protocol for Metadata Harvesting*).

### 3.2.2 Metode Pengumpulan Data

Pengumpulan data dilakukan secara otomatis menggunakan teknik *harvesting* dengan library Sickle (Python) dan cloudscraper untuk mengatasi proteksi anti-bot pada server.

[Tabel 3.1: Atribut Metadata yang Dikumpulkan](placeholder/tabel_3_1_atribut_metadata.md)

### 3.2.3 Populasi dan Sampel

- **Populasi**: Seluruh metadata tugas akhir di repositori Universitas Widyatama
- **Sampel**: Seluruh records dari koleksi *Final Assignment (Bachelor and Vocational Degree)*
- **Metode Sampling**: *Purposive sampling* - dokumen dengan abstrak lengkap berbahasa Indonesia

---

## 3.3 Pembersihan dan Validasi Data

### 3.3.1 Kriteria Inklusi

1. Memiliki abstrak dengan panjang minimal yang memadai
2. Abstrak ditulis dalam bahasa Indonesia
3. Tidak memiliki duplikasi berdasarkan identifier

### 3.3.2 Proses Pembersihan Data

1. Penanganan *missing values*
2. Deduplikasi
3. Normalisasi teks
4. Validasi bahasa

---

## 3.4 Preprocessing Teks

### 3.4.1 Tahapan Preprocessing

[Gambar 3.2: Diagram Alur Preprocessing Teks](placeholder/gambar_3_2_diagram_preprocessing.md)

### 3.4.2 Case Folding

Konversi teks ke huruf kecil (*lowercase*).

### 3.4.3 Pembersihan Karakter Khusus

Penghapusan tanda baca, angka, dan karakter non-alfanumerik.

### 3.4.4 Tokenisasi

Tokenisasi menggunakan library nlp-id untuk bahasa Indonesia.

### 3.4.5 Stopword Removal

Penghapusan kata fungsional menggunakan gabungan stopwords dari:
- NLTK Indonesian Stopwords
- NLTK English Stopwords
- PySastrawi Stopwords
- Custom Stopwords (domain-specific)

[Tabel 3.2: Contoh Custom Stopwords berdasarkan Kategori](placeholder/tabel_3_2_custom_stopwords.md)

### 3.4.6 Phrase Detection (Bigrams)

Deteksi frasa dua kata menggunakan model Gensim Phrases dengan parameter minimum count dan threshold tertentu.

### 3.4.7 Filtering Token

Filter berdasarkan panjang kata (minimum 3, maksimum 50 karakter) dan panjang dokumen (minimum 30 tokens).

---

## 3.5 Pemodelan Topik dengan LDA

### 3.5.1 Algoritma Latent Dirichlet Allocation

Penjelasan konsep LDA sebagai model probabilistik generatif untuk koleksi teks, di mana:
- Setiap dokumen direpresentasikan sebagai campuran probabilistik dari topik
- Setiap topik direpresentasikan sebagai campuran probabilistik dari kata

[Gambar 3.3: Ilustrasi Konsep Model LDA](placeholder/gambar_3_3_ilustrasi_lda.md)

### 3.5.2 Pembuatan Dictionary dan Corpus

1. **Dictionary**: Mapping kata ke ID numerik dengan filtering berdasarkan frekuensi dokumen
2. **Corpus**: Representasi Bag-of-Words (BoW) dari setiap dokumen

### 3.5.3 Hyperparameter Model LDA

[Tabel 3.3: Konfigurasi Hyperparameter LDA](placeholder/tabel_3_3_hyperparameter_lda.md)

### 3.5.4 Metode Optimasi Jumlah Topik

Pencarian jumlah topik optimal menggunakan metode *grid search* dengan rentang nilai tertentu dan evaluasi menggunakan Coherence Score.

---

## 3.6 Metode Evaluasi Model

### 3.6.1 Topic Coherence (C_v)

Coherence score digunakan untuk mengukur koherensi semantik kata-kata dalam setiap topik. Metrik ini dipilih karena berkorelasi tinggi dengan penilaian manusia terhadap interpretabilitas topik.

[Tabel 3.4: Kriteria Interpretasi Nilai Coherence Score](placeholder/tabel_3_4_kriteria_coherence.md)

---

## 3.7 Rancangan Visualisasi

### 3.7.1 Metode Visualisasi

1. Word Cloud untuk kata-kata dominan per topik
2. pyLDAvis untuk visualisasi interaktif distribusi topik
3. Bar Chart untuk top-N kata per topik
4. Heatmap untuk matriks topik-dokumen
5. Line Chart untuk analisis tren topik

### 3.7.2 Dashboard Interaktif

Dashboard dibangun menggunakan Streamlit untuk eksplorasi hasil pemodelan topik.

[Tabel 3.5: Rancangan Halaman Dashboard](placeholder/tabel_3_5_rancangan_dashboard.md)

---

## 3.8 Tools dan Teknologi

### 3.8.1 Bahasa Pemrograman

Python 3.12

### 3.8.2 Library Utama

[Tabel 3.6: Library Utama yang Digunakan](placeholder/tabel_3_6_library_utama.md)

---

## 3.9 Diagram Alur Sistem

[Gambar 3.4: Diagram Alur Sistem Keseluruhan](placeholder/gambar_3_4_diagram_alur_sistem.md)
