# JudiScope

Studi perbandingan **Artificial Neural Network (ANN), Support Vector Machine (SVM), dan Multinomial Naive Bayes** untuk mendeteksi komentar promosi judi online di YouTube.

Proyek tugas besar pengganti UTS ini mencakup EDA, cleaning teks, augmentasi sintetis, training, evaluasi, dan analisis kesalahan. Eksperimen Naive Bayes (Atharik), ANN (Bagas), dan SVM (Aidan) tersedia dalam satu workflow dengan dataset dan pembagian yang sama. Hasilnya menjadi bahan perbandingan dalam paper.

## Mulai dari sini

1. Baca [konteks proyek](KONTEKS_PROYEK.md) untuk tujuan penelitian dan arah paper.
2. Baca [panduan dataset tim](PANDUAN_DATASET_TIM.md) sebelum mengerjakan ANN atau SVM.
3. Ikuti urutan notebook untuk memeriksa atau mereproduksi hasil.

## Dataset

Sumber awal dataset, berdasarkan informasi anggota kelompok, adalah **[Judi Online di Kaggle — `yaemico/judionline`](https://www.kaggle.com/datasets/yaemico/judionline)**. Data awal merupakan data sekunder; cleaning dan augmentasi dilakukan dalam proyek ini. Atribusi serta rincian yang belum terverifikasi dicatat di [dataset/README.md](dataset/README.md).

| Tahap | Jumlah | Keterangan |
| --- | ---: | --- |
| File mentah lokal | 10.230 | Komentar dari 32 video dan 18 channel |
| Hasil cleaning | 8.934 | Teks asli unik; duplikat disisihkan dan konflik label dipisahkan |
| Tambahan sintetis | 1.066 | Komentar buatan berbasis template, khusus skenario latihan |
| Gabungan | 10.000 | 8.934 asli bersih + 1.066 sintetis |

Input model adalah `komentar_processed`, dengan target `label`. Pemetaan kerja saat ini: `0 = nonjudi`, `1 = promosi judi`. Arti label dan metode anotasi dari sumber masih perlu dikonfirmasi. **10.000 record gabungan bukan 10.000 komentar YouTube asli.**

## Struktur repository

```text
JudiScope/
├── README.md
├── KONTEKS_PROYEK.md
├── PANDUAN_DATASET_TIM.md
├── requirements.txt
├── 01_eda_cleaning.ipynb
├── 02_dataset_generator.ipynb
├── 03_bayesian_naive_bayes.ipynb
├── 04_ann.ipynb
├── 05_svm.ipynb
├── dataset/
│   ├── README.md
│   └── data_komentar_dengan_prediksi - data_komentar_dengan_prediksi(2).csv
└── outputs/
    ├── cleaning/       # Data bersih, review, audit, dan ringkasan
    ├── augmentation/   # Data sintetis, gabungan, dan konfigurasi generator
    ├── bayesian/       # Model, manifest split, skor, prediksi, dan grafik
    ├── ann/            # Model ANN, skor, prediksi, bobot proksi, dan grafik
    └── svm/            # Skor, prediksi, koefisien, grafik, dan model SVM lokal
```

Folder sumber menggunakan nama **`dataset`**, seluruhnya huruf kecil.

## Instalasi

Gunakan Python **3.11 atau lebih baru**. `requirements.txt` berisi dependensi bersama untuk notebook 01–05. Rentang versinya mendukung workflow bersama; file ini bukan lock lingkungan seluruh hasil historis. Versi aktual tersimpan dalam `outputs/<metode>/experiment_summary.json`.

| Hasil tersimpan | Python | scikit-learn |
| --- | --- | --- |
| Naive Bayes | 3.14.3 | 1.8.0 |
| ANN | 3.15.0rc2 | 1.9.1 |
| SVM | 3.11.9 | 1.4.2 |

Untuk mereproduksi hasil historis secara persis atau memuat model joblib, gunakan versi yang sesuai dengan ringkasan metode tersebut. Menjalankan ulang pada versi berbeda dapat mengubah hasil numerik.

```bash
git clone https://github.com/atharikputra/JudiScope.git
cd JudiScope
python -m venv .venv
```

Aktifkan environment pada Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Pada macOS/Linux:

```bash
source .venv/bin/activate
```

Kemudian pasang dependensi dan buka JupyterLab:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Buka notebook dari folder utama repository. Alternatifnya, buka dengan VS Code dan pilih kernel Python dari `.venv`.

## Urutan notebook

| Notebook | Isi | Hasil utama |
| --- | --- | --- |
| [01 — EDA dan cleaning](01_eda_cleaning.ipynb) | Struktur data, distribusi label, panjang komentar, duplikat, konflik, lalu cleaning | `outputs/cleaning/komentar_cleaned.csv` |
| [02 — Generator dataset](02_dataset_generator.ipynb) | Konfigurasi target dan seed, generator sintetis, preview, verifikasi, ekspor | `outputs/augmentation/komentar_synthetic_only.csv` dan CSV gabungan |
| [03 — Bayesian](03_bayesian_naive_bayes.ipynb) | Pipeline TF-IDF + MultinomialNB, tuning, perbandingan augmentasi, evaluasi, analisis kesalahan | `outputs/bayesian/` |
| [04 — ANN](04_ann.ipynb) | Pipeline TF-IDF + MLPClassifier, tuning arsitektur, evaluasi perbandingan, analisis kesalahan | `outputs/ann/` |
| [05 — SVM](05_svm.ipynb) | Pipeline TF-IDF + LinearSVC, tuning `C`, evaluasi dua skenario, analisis kesalahan dan fitur | `outputs/svm/` |

Jalankan sel dari atas ke bawah. File hasil sudah disertakan sehingga anggota tim dapat memakai manifest dan dataset tanpa training ulang Bayesian. Menjalankan ulang sel ekspor akan menimpa hasil dengan nama yang sama; file sumber mentah tidak ditimpa.

Notebook 03 dijalankan terlebih dahulu saat membuat manifest dari awal. Setelah data bersih, sintetis, dan manifest tersedia, notebook 04 dan 05 dapat dijalankan secara terpisah. SVM membaca langsung file JudiScope dan tidak membutuhkan repository `svm-classifier` atau langkah impor dataset tambahan.

Untuk mengeksekusi SVM lewat terminal setelah dependensi terpasang:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace 05_svm.ipynb
```

Cleaning mempertahankan angka, tanda baca, dan emoji; menerapkan NFKC, huruf kecil, perapian karakter tak terlihat/spasi, serta penanda URL. Baseline tidak memakai stemming atau stopword removal.

## Protokol perbandingan ANN, SVM, dan Bayesian

Gunakan pembagian dari **`outputs/bayesian/original_split_manifest.csv`**, dengan `source_row` sebagai kunci. Jangan membuat split baru per algoritma atau membagi CSV gabungan 10.000 secara acak.

| Tahap | Asli saja | Asli + sintetis |
| --- | ---: | ---: |
| Train untuk tuning | 5.360 | 6.426 (5.360 asli + 1.066 sintetis) |
| Validation | 1.787 asli | 1.787 asli yang sama |
| Refit setelah seleksi | 7.147 asli | 8.213 (7.147 asli + 1.066 sintetis) |
| Test akhir | 1.787 asli | 1.787 asli yang sama |

Parameter dan skenario dipilih berdasarkan **F1 kelas `1` pada validation**, dengan macro-F1 sebagai pemecah seri. TF-IDF hanya belajar dari bagian latihan tiap kandidat. Setelah pilihan dikunci, pipeline dilatih ulang pada train + validation dan diuji pada test asli. Komentar sintetis tidak masuk validation/test.

Bandingkan accuracy, precision kelas `1`, recall kelas `1`, F1 kelas `1`, macro-F1, confusion matrix, waktu latihan, dan contoh kesalahan. Detail kode pembacaan data serta pengaturan fitur tersedia di [panduan tim](PANDUAN_DATASET_TIM.md).

## Hasil awal Bayesian

Evaluasi pada **1.787 komentar test asli**:

| Skenario | Accuracy | Precision kelas `1` | Recall kelas `1` | F1 kelas `1` |
| --- | ---: | ---: | ---: | ---: |
| Asli saja | **97,87%** | **95,50%** | **98,12%** | **96,79%** |
| Asli + sintetis | 96,81% | 93,99% | 96,40% | 95,18% |

Kedua konfigurasi terbaik memakai TF-IDF karakter `char_wb` 3–5 gram dan MultinomialNB `alpha=0.1`. Skenario asli saja dipilih dari validation sebelum test. Pada eksperimen ini, tambahan sintetis menurunkan performa. Hasil metode lain tersedia di folder keluarannya masing-masing.

![Confusion matrix Bayesian tanpa dan dengan data sintetis](outputs/bayesian/confusion_matrices.png)

Metrik lengkap ada di [test_metrics.csv](outputs/bayesian/test_metrics.csv), konfigurasi di [experiment_summary.json](outputs/bayesian/experiment_summary.json), dan prediksi tiap record di [test_predictions.csv](outputs/bayesian/test_predictions.csv).

## Eksperimen SVM

[05_svm.ipynb](05_svm.ipynb) memuat pekerjaan Aidan: TF-IDF kata 1–2 gram atau karakter 3–5 gram, dengan `LinearSVC` dan `C = 0.1, 1, 10`. Pemilihan fitur, parameter, dan skenario memakai validation; kemudian pipeline baru dilatih pada development dan diuji pada test asli.

| Data latihan | Accuracy | Precision kelas `1` | Recall kelas `1` | F1 kelas `1` |
| --- | ---: | ---: | ---: | ---: |
| Asli saja | 98,77% | 99,65% | 96,58% | 98,09% |
| Asli + sintetis | 98,71% | 99,47% | 96,58% | 98,00% |

Keduanya memilih karakter 3–5 gram dan `C=10`. Skenario asli saja dipilih berdasarkan validation. Model tersebut menghasilkan 2 false positive dan 20 false negative pada 1.787 komentar test. Notebook membahas hasil SVM; uji statistik lintasmetode dikelola dalam analisis paper.

| File di `outputs/svm/` | Isi |
| --- | --- |
| `validation_results.csv` | Seluruh 12 kandidat, parameter, konvergensi, dan metrik validation |
| `test_metrics.csv` | Metrik dua skenario dan majority baseline, skala 0–1 |
| `test_predictions.csv` | Satu baris per `source_row`, label, teks, prediksi, dan margin kedua skenario |
| `selected_model_errors.csv` | Kesalahan model yang dipilih melalui validation |
| `selected_model_feature_weights.csv` | Fitur dan koefisien SVM model terpilih |
| `experiment_summary.json` | Konfigurasi, hash input, versi, hasil, dan pemeriksaan |
| `*.png` | Grafik validation, confusion matrix, metrik test, dan contoh koefisien |

Kolom `decision_score_original_only` dan `decision_score_with_synthetic` adalah margin keputusan SVM, **bukan probabilitas**. Input, pemilihan parameter, prediksi, margin, dan koefisien hasil integrasi telah diperiksa terhadap run SVM sebelumnya. Hasil numeriknya cocok.

Seluruh sel kode SVM telah dijalankan dan output disimpan melalui IPython berurutan pada lingkungan yang tersedia. Eksekusi melalui transport kernel Jupyter belum diuji di lingkungan tersebut; gunakan Run All di Jupyter/VS Code atau perintah di atas untuk menjalankannya melalui kernel lokal.

## Model tersimpan

`outputs/bayesian/` menyertakan pipeline `.joblib` untuk kedua skenario. Pipeline mencakup TF-IDF dan classifier, tetapi menerima **teks yang sudah dibersihkan**. Untuk masukan mentah, gunakan fungsi `clean_comment` pada notebook Bayesian terlebih dahulu. Muat file joblib hanya dari sumber tepercaya.

SVM menyimpan `outputs/svm/svm_original_only.joblib` dan `svm_with_synthetic.joblib` ketika notebook dijalankan. Model SVM diabaikan Git; notebook, CSV, JSON, dan PNG tetap disertakan. Fungsi cleaning dan demo prediksi SVM ada langsung dalam notebook 05. Model Bayesian dan ANN yang sudah dilacak tetap mengikuti struktur asal repository.

Sel kode telah dijalankan berurutan lewat Python, hasil tabel/grafik disimpan dalam notebook, dan kesamaan prediksi model setelah disimpan/dimuat ulang sudah diperiksa. Eksekusi melalui kernel Jupyter belum diverifikasi pada lingkungan pengerjaan awal.

## Batas penelitian dan arah paper

Split saat ini acak pada teks unik, bukan holdout video, kampanye, atau periode. Kemiripan template dapat tersebar pada bagian berbeda; skor belum membuktikan generalisasi ke sumber baru. Kualitas anotasi, versi sumber Kaggle, dan rincian lisensi dataset belum terverifikasi. Penanda demo serta template sintetis juga dapat menjadi petunjuk buatan bagi model.

Hasil yang ditargetkan adalah kode eksperimen, tabel perbandingan, pembahasan kesalahan, paper, dan presentasi. Usulan judul paper:

> Perbandingan ANN, SVM, dan Multinomial Naive Bayes untuk Deteksi Komentar Promosi Judi Online di YouTube

Referensi sumber dataset harus dicantumkan pada paper; komentar sintetis dilaporkan terpisah sebagai buatan kelompok. Repository ini tidak menetapkan lisensi baru untuk dataset sumber.
