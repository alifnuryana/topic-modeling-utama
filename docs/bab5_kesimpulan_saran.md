# BAB 5: KESIMPULAN DAN SARAN

## 5.1 Kesimpulan

Berdasarkan hasil penelitian yang telah dilakukan tentang penerapan *topic modeling* menggunakan algoritma *Latent Dirichlet Allocation* (LDA) untuk analisis metadata tugas akhir di repositori Universitas Widyatama, dapat disimpulkan beberapa hal sebagai berikut:

### 5.1.1 Pengumpulan dan Pengolahan Data

1. **Pengumpulan data** metadata tugas akhir melalui protokol OAI-PMH berhasil dilakukan dengan mengumpulkan **12.647 records** dari repositori institusional Universitas Widyatama. Setelah melalui proses pembersihan data yang meliputi deduplikasi, penanganan *missing values*, dan filtering berdasarkan panjang abstrak, diperoleh **12.077 records** bersih (95,5% data dipertahankan).

2. **Preprocessing teks** bahasa Indonesia berhasil diimplementasikan dengan menggabungkan berbagai teknik: *case folding*, pembersihan karakter khusus, tokenisasi menggunakan library nlp-id, *stopword removal* dari berbagai sumber (NLTK Indonesian, NLTK English, PySastrawi, dan *custom stopwords* domain-specific), serta deteksi frasa menggunakan Gensim Phrases. Proses ini menghasilkan **11.979 dokumen** dengan rata-rata **71,9 tokens per dokumen**.

### 5.1.2 Pemodelan Topik

3. **Optimasi jumlah topik** dilakukan dengan menggunakan *coherence score* (C_v) pada rentang 4-14 topik. Hasil evaluasi menunjukkan bahwa **12 topik** merupakan jumlah optimal dengan *coherence score* sebesar **0,5546**, yang termasuk dalam kategori "baik" berdasarkan kriteria interpretasi standar (0,5-0,6).

4. **Model LDA** yang dikembangkan berhasil mengidentifikasi **12 topik utama** penelitian tugas akhir di Universitas Widyatama dengan interpretabilitas yang tinggi. Penggunaan *bigram* (frasa dua kata) seperti "kinerja_karyawan", "audit_internal", dan "keputusan_pembelian" secara signifikan meningkatkan kualitas interpretasi topik.

### 5.1.3 Karakteristik Topik Penelitian

5. **Topik dominan** yang teridentifikasi adalah:
   - **Manajemen Pemasaran** (12,5% - 1.495 dokumen) dengan fokus pada perilaku konsumen dan keputusan pembelian
   - **Auditing & Pengendalian Internal** (12,4% - 1.483 dokumen) dengan fokus pada *good corporate governance*
   - **Manajemen SDM** (12,1% - 1.448 dokumen) dengan fokus pada kinerja karyawan dan produktivitas

6. **Distribusi topik** menunjukkan bahwa bidang Bisnis (Manajemen, Akuntansi, dan Keuangan) mendominasi sekitar **70%** dari total dokumen, yang mencerminkan karakteristik program studi yang ada di Universitas Widyatama. Kehadiran topik Teknik Informatika (6,7%) dan Industri Kreatif (3,8%) menunjukkan diversifikasi program studi di institusi.

### 5.1.4 Dashboard Interaktif

7. **Dashboard Streamlit** berhasil dikembangkan dengan **8 halaman interaktif** yang memungkinkan eksplorasi komprehensif terhadap hasil pemodelan topik. Fitur-fitur utama meliputi: eksplorasi topik dengan *word cloud*, peramban dokumen dengan filter multi-kriteria, pencarian kemiripan dokumen, analisis tren temporal, perbandingan topik, dan pelabelan topik *custom*.

### 5.1.5 Kontribusi Penelitian

Penelitian ini memberikan kontribusi pada:
- **Validasi penerapan LDA** untuk analisis teks akademik berbahasa Indonesia
- **Pemetaan lanskap penelitian** institusional yang dapat digunakan sebagai dasar pengambilan keputusan strategis
- **Pengembangan *pipeline* otomatis** untuk *topic modeling* yang dapat direplikasi untuk repositori akademik lain
- **Prototipe dashboard interaktif** yang memfasilitasi eksplorasi hasil penelitian oleh berbagai *stakeholder*

---

## 5.2 Saran

Berdasarkan hasil penelitian dan keterbatasan yang ditemukan, berikut adalah saran untuk pengembangan penelitian selanjutnya:

### 5.2.1 Saran untuk Institusi

1. **Implementasi sistem produksi**: Dashboard yang dikembangkan direkomendasikan untuk diintegrasikan ke dalam sistem repositori institusional sebagai layanan nilai tambah bagi mahasiswa, dosen, dan pustakawan dalam mengeksplorasi tren penelitian.

2. **Pembaruan berkala**: Disarankan untuk menjalankan proses *topic modeling* secara berkala (misalnya setiap semester) agar pemetaan topik penelitian tetap *up-to-date* seiring bertambahnya dokumen baru di repositori.

3. **Pengembangan kurikulum**: Peta topik penelitian yang dihasilkan dapat digunakan sebagai masukan untuk evaluasi dan pengembangan kurikulum program studi, dengan memperhatikan tren topik yang naik atau turun dari waktu ke waktu.

### 5.2.2 Saran untuk Penelitian Selanjutnya

4. **Ekspansi sumber data**: Penelitian selanjutnya dapat memperluas cakupan data dengan mengikutsertakan repositori dari beberapa universitas untuk mendapatkan gambaran yang lebih komprehensif tentang lanskap penelitian akademik Indonesia.

5. **Analisis *full-text***: Penggunaan teks lengkap tugas akhir (bukan hanya abstrak) dapat meningkatkan kedalaman dan akurasi pemodelan topik, meskipun akan memerlukan sumber daya komputasi yang lebih besar.

6. **Eksplorasi algoritma lain**: Perbandingan dengan metode *topic modeling* lain seperti **BERTopic**, **Top2Vec**, atau **Neural LDA** dapat dilakukan untuk mengevaluasi apakah algoritma berbasis *transformer* menghasilkan topik dengan kualitas yang lebih baik untuk teks bahasa Indonesia.

7. **Peningkatan NLP bahasa Indonesia**: Pengembangan atau penyempurnaan *tools* pemrosesan bahasa Indonesia (seperti *stemmer*, *lemmatizer*, dan *named entity recognition*) dapat meningkatkan kualitas *preprocessing* dan hasil akhir pemodelan topik.

8. **Integrasi sistem rekomendasi**: Berdasarkan hasil pemodelan topik, dapat dikembangkan sistem rekomendasi literatur yang membantu mahasiswa menemukan tugas akhir yang relevan dengan topik penelitian mereka.

9. **Analisis sentimen dan opini**: Kombinasi *topic modeling* dengan analisis sentimen dapat memberikan wawasan tambahan tentang bagaimana topik-topik tertentu dibahas dalam konteks positif atau negatif.

### 5.2.3 Saran Teknis

10. **Optimasi performa**: Implementasi *caching* dan *lazy loading* pada dashboard dapat meningkatkan performa, terutama ketika jumlah dokumen bertambah secara signifikan.

11. **Dukungan multi-bahasa**: Pengembangan sistem yang dapat menangani dokumen dalam berbagai bahasa (Indonesia, Inggris, atau campuran) akan meningkatkan fleksibilitas dan jangkauan analisis.

12. **Dokumentasi dan *reproducibility***: Pembuatan dokumentasi yang lengkap dan kontainerisasi sistem (menggunakan Docker) akan memudahkan replikasi dan adopsi oleh institusi lain.
