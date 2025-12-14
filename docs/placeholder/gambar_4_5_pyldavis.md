# Placeholder untuk Gambar 4.5

## Informasi Gambar
- **Judul**: Visualisasi pyLDAvis - Intertopic Distance Map
- **Lokasi File Aktual**: `outputs/pyldavis.html` (interaktif)
- **Deskripsi**: Visualisasi interaktif yang menampilkan posisi relatif 12 topik dalam ruang 2D berdasarkan Principal Component Analysis

## Komponen Visualisasi

### Panel Kiri: Intertopic Distance Map
- 12 lingkaran merepresentasikan 12 topik
- Ukuran lingkaran proporsional dengan prevalensi topik
- Jarak antar lingkaran merepresentasikan kesamaan topik
- Sumbu: PC1 dan PC2 (Principal Components)

### Panel Kanan: Top 30 Most Relevant Terms
- Bar chart menampilkan kata-kata paling relevan untuk topik terpilih
- Warna merah: frekuensi kata dalam topik terpilih
- Warna biru: frekuensi keseluruhan kata dalam corpus

## Observasi Cluster

### Cluster 1: Akuntansi-Keuangan
- Topik 0 (Akuntansi Biaya)
- Topik 4 (Profitabilitas)
- Topik 7 (Perbankan)
- Topik 10 (Pasar Modal)

### Cluster 2: SDM-Organisasi
- Topik 1 (Manajemen SDM)
- Topik 6 (Kepemimpinan)
- Topik 9 (Sistem Informasi Manajemen)

### Topik Tersebar
- Topik 3 (Teknik Informatika) - berbeda karena fokus teknologi
- Topik 11 (Pemasaran) - unik karena fokus konsumen

## Spesifikasi
- Format: HTML interaktif dengan D3.js
- Dapat di-export sebagai screenshot statis untuk dokumen
