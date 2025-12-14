# Tabel 3.3: Konfigurasi Hyperparameter LDA

| Parameter | Nilai | Deskripsi |
|-----------|-------|-----------|
| `num_topics` | Dicari optimal | Jumlah topik |
| `passes` | 20 | Jumlah iterasi melalui korpus |
| `iterations` | 400 | Maksimum iterasi per dokumen |
| `chunksize` | 3.000 | Dokumen per batch training |
| `alpha` | symmetric | Parameter prior distribusi topik-dokumen |
| `eta` | auto | Parameter prior distribusi kata-topik |
| `workers` | 6 | Jumlah thread paralel |
| `random_state` | 42 | Seed untuk reprodusibilitas |
