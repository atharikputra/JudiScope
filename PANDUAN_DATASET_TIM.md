# Panduan dataset untuk eksperimen ANN, SVM, dan Bayesian

Gunakan data, label, dan pembagian train/validation/test yang sama agar hasil tiga algoritma bisa dibandingkan. Notebook Bayesian membuat manifest pembagian; [04_ann.ipynb](04_ann.ipynb) dan [05_svm.ipynb](05_svm.ipynb) membaca manifest tersebut secara langsung. Hasil masing-masing tersedia di `outputs/bayesian/`, `outputs/ann/`, dan `outputs/svm/`.

Sumber awal dataset, berdasarkan informasi anggota kelompok: **[Judi Online — Kaggle (`yaemico/judionline`)](https://www.kaggle.com/datasets/yaemico/judionline)**. Gunakan salinan lokal yang sama, bukan mengunduh versi lain lalu membuat split sendiri. Rincian atribusi dan hal yang belum terverifikasi dicatat di [dataset/README.md](dataset/README.md).

## 1. File yang dibagikan ke anggota kelompok

Salin file berikut beserta struktur foldernya. Path dalam panduan ini dihitung dari folder utama proyek.

| File | Isi | Pemakaian |
| --- | --- | --- |
| `outputs/cleaning/komentar_cleaned.csv` | 8.934 komentar asli setelah cleaning dan deduplikasi | Dataset utama untuk semua algoritma |
| `outputs/bayesian/original_split_manifest.csv` | `source_row` dan bagian data: `train`, `validation`, atau `test` | Pembagian yang wajib diikuti; jangan membuat split baru |
| `outputs/augmentation/komentar_synthetic_only.csv` | 1.066 komentar sintetis berlabel `1` | Tambahan untuk latihan pada eksperimen augmentasi |
| `outputs/bayesian/experiment_summary.json` | Konfigurasi, hash input, dan hasil Bayesian | Referensi eksperimen dan reproduksi |

`outputs/augmentation/komentar_10000_with_synthetic.csv` merupakan gabungan asli + sintetis. Jangan langsung menjalankan `train_test_split` pada file ini: sintetis akan berpotensi masuk validation/test. Untuk modeling, gunakan tiga CSV terpisah pada tabel di atas.

File CSV mentah dalam `dataset/` tetap disimpan sebagai sumber. Gunakan hasil cleaning yang sudah disepakati untuk perbandingan ini; jangan masing-masing melakukan cleaning ulang yang berbeda.

## 2. Kolom input dan target

| Kolom | Peran |
| --- | --- |
| `komentar_processed` | **X**: teks hasil cleaning yang dimasukkan ke TF-IDF |
| `label` | **y**: target klasifikasi |
| `source_row` | ID untuk mencocokkan data asli dengan manifest; bukan fitur |
| `komentar` | Teks asli untuk membaca contoh dan menganalisis kesalahan |

Pemetaan kerja saat ini adalah `0 = nonjudi` dan `1 = promosi judi`. Arti label serta cara anotasi dari sumber belum terkonfirmasi; tulis keterbatasan ini dalam laporan.

Jangan memakai label, prediksi model lama, metadata video/author/tanggal, `duplicate_count`, ID, atau penanda sintetis sebagai fitur baseline. Informasi tersebut hanya membantu audit atau pelacakan data.

## 3. Dua skenario eksperimen

| Tahap | Tanpa sintetis (`original_only`) | Dengan sintetis (`with_synthetic`) |
| --- | ---: | ---: |
| Train saat memilih parameter | 5.360 asli | 5.360 asli + 1.066 sintetis = 6.426 |
| Validation untuk memilih parameter | 1.787 asli | 1.787 asli yang sama |
| Refit setelah parameter dikunci | 7.147 asli (train + validation) | 7.147 asli + 1.066 sintetis = 8.213 |
| Test untuk evaluasi akhir | 1.787 asli | 1.787 asli yang sama |

Proporsi data asli adalah sekitar 60%/20%/20%, dengan seed `42` dan stratifikasi label. Tetap baca manifest, bukan membuat ulang pembagian berdasarkan persentase atau seed saja.

Urutan kerja:

1. Baca dataset asli, manifest, dan sintetis.
2. Bentuk bagian data menggunakan `source_row`.
3. Latih kandidat pada train; untuk skenario augmentasi, tambahkan sintetis hanya di bagian ini.
4. Pilih parameter dari **F1 kelas `1` pada validation**. Gunakan macro-F1 sebagai pemecah seri, mengikuti Bayesian.
5. Kunci pilihan parameter dan skenario sebelum melihat skor test.
6. Buat model/pipeline baru dan latih ulang pada train + validation; tambahkan sintetis untuk skenario augmentasi.
7. Uji kedua skenario yang telah ditetapkan pada test asli yang sama. Jangan mengganti parameter berdasarkan hasil test.

Kalau hanya sempat menjalankan satu skenario, gunakan **tanpa sintetis** sebagai baseline bersama dan jelaskan cakupan eksperimennya.

## 4. Contoh kode membaca dataset dan mengikuti manifest

Jalankan dari folder utama proyek. Jika notebook berada di subfolder `notebooks`, kode di bawah mencoba folder induknya. Sesuaikan `PROJECT_DIR` bila struktur folder berbeda.

```python
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path.cwd().resolve()
if not (PROJECT_DIR / "outputs" / "cleaning").is_dir():
    PROJECT_DIR = PROJECT_DIR.parent

original = pd.read_csv(
    PROJECT_DIR / "outputs/cleaning/komentar_cleaned.csv",
    dtype=str, keep_default_na=False, encoding="utf-8-sig",
)
manifest = pd.read_csv(
    PROJECT_DIR / "outputs/bayesian/original_split_manifest.csv",
    dtype=str, keep_default_na=False,
)
synthetic = pd.read_csv(
    PROJECT_DIR / "outputs/augmentation/komentar_synthetic_only.csv",
    dtype=str, keep_default_na=False, encoding="utf-8-sig",
)

assert original["source_row"].is_unique
assert manifest["source_row"].is_unique
assert set(original["source_row"]) == set(manifest["source_row"])
assert set(manifest["split"]) == {"train", "validation", "test"}
assert original["label"].isin(["0", "1"]).all()
assert synthetic["label"].eq("1").all()
assert synthetic["is_synthetic"].str.lower().eq("true").all()
assert not original["komentar_processed"].duplicated().any()
assert not synthetic["komentar_processed"].duplicated().any()
assert set(original["komentar_processed"]).isdisjoint(
    synthetic["komentar_processed"]
)

# Urutan record mengikuti manifest Bayesian, bukan nomor indeks CSV.
data = manifest.merge(original, on="source_row", how="left", validate="one_to_one")
data["label"] = data["label"].astype(int)
synthetic["label"] = synthetic["label"].astype(int)

train = data.loc[data["split"].eq("train")].copy()
validation = data.loc[data["split"].eq("validation")].copy()
test = data.loc[data["split"].eq("test")].copy()

assert (len(train), len(validation), len(test)) == (5360, 1787, 1787)
assert len(synthetic) == 1066

model_columns = ["komentar_processed", "label"]
train_original = train[model_columns].copy()
train_augmented = pd.concat(
    [train_original, synthetic[model_columns]], ignore_index=True
)

X_train = train_original["komentar_processed"]
y_train = train_original["label"]
X_train_augmented = train_augmented["komentar_processed"]
y_train_augmented = train_augmented["label"]
X_validation = validation["komentar_processed"]
y_validation = validation["label"]
X_test = test["komentar_processed"]
y_test = test["label"]

# Data refit: baru dipakai setelah pemilihan parameter selesai.
development_original = pd.concat(
    [train[model_columns], validation[model_columns]], ignore_index=True
)
development_augmented = pd.concat(
    [development_original, synthetic[model_columns]], ignore_index=True
)

print("Train asli:", len(train_original))
print("Train + sintetis:", len(train_augmented))
print("Validation asli:", len(validation))
print("Test asli:", len(test))
```

Jika assertion gagal, periksa versi file yang dibagikan. Jangan menghapus assertion lalu meneruskan dengan dataset yang berbeda. `source_row` adalah identitas record asli; jangan dinomori ulang.

## 5. TF-IDF dan pelatihan model

Gunakan `Pipeline` yang berisi TF-IDF dan classifier. Setiap kandidat belajar kosakata serta IDF hanya dari data latihan kandidat tersebut. Validation dan test cukup ditransformasi menggunakan TF-IDF yang sudah dipelajari. Saat refit, buat pipeline baru dan fit pada data development sesuai skenarionya.

Grid fitur bersama Bayesian, ANN, dan SVM:

| Pengaturan | Nilai |
| --- | --- |
| Kandidat A | `analyzer="word"`, `ngram_range=(1, 2)` |
| Kandidat B | `analyzer="char_wb"`, `ngram_range=(3, 5)` |
| Keduanya | `min_df=2`, `max_features=40000`, `sublinear_tf=True`, `lowercase=False` |

Ketiga notebook mencoba dua kandidat fitur tersebut dengan aturan seleksi validation yang sama. Parameter classifier menyesuaikan algoritmanya: Bayesian menguji `alpha`, ANN menguji jumlah unit tersembunyi, dan SVM menguji `C`. Catat jumlah kandidat dan pengaturan terbaik; jumlah kandidat bukan ukuran biaya komputasi yang setara antaralgoritma.

ANN dan SVM tidak perlu memakai nilai `alpha` milik Naive Bayes. Untuk algoritma yang memiliki proses acak, set `random_state=42` jika tersedia. Jika ANN membutuhkan representasi yang lebih kecil atau konversi sparse ke dense, pertimbangkan kebutuhan memori, pelajari transformasi tambahan hanya dari train, dan dokumentasikan perbedaannya.

## 6. Hasil yang dikumpulkan setiap anggota

Untuk setiap skenario, simpan:

- Algoritma, parameter, pengaturan TF-IDF, seed, dan versi library.
- Hasil validation yang menjadi dasar pemilihan parameter.
- Accuracy, precision kelas `1`, recall kelas `1`, F1 kelas `1`, dan macro-F1 pada test.
- Confusion matrix dengan urutan label `[0, 1]`.
- Waktu fit dan jumlah record latihan; bedakan waktu tuning dari waktu refit.
- Prediksi setiap komentar test, disertai `source_row` dan label rujukannya.
- Contoh false positive dan false negative untuk pembahasan.

Keluaran notebook SVM mengikuti susunan Bayesian/ANN: `validation_results.csv`, `test_metrics.csv`, `test_predictions.csv`, `selected_model_errors.csv`, `selected_model_feature_weights.csv`, `experiment_summary.json`, serta grafik. Prediksi kedua skenario berada pada kolom terpisah dalam satu baris per komentar. SVM menyimpan margin pada kolom `decision_score_<skenario>`; angka ini bukan probabilitas seperti keluaran `predict_proba`.

Laporkan metrik konsisten: nilai 0–1 atau persentase dengan format yang sama. Precision/recall/F1 kelas `1` dapat dihitung dengan `pos_label=1` dan `zero_division=0`; macro-F1 memakai `average="macro"`.

## 7. Batas kesimpulan

Komentar sintetis adalah data buatan berbasis template, bukan tambahan observasi YouTube asli. Sebutkan jumlahnya secara terpisah. Jangan menyembunyikan jika augmentasi menurunkan performa.

Split saat ini memisahkan teks unik secara acak. Kemiripan template, nama situs, atau video masih bisa muncul di beberapa bagian. Hasil ini belum membuktikan kemampuan pada video, kampanye, atau situs yang benar-benar baru. Untuk evaluasi lanjutan, pertimbangkan pemisahan berdasarkan video/kampanye dan pemeriksaan ulang label.

Panduan ini tidak membutuhkan training ulang Bayesian. Manifest, hasil cleaning, dan dataset sintetis yang sudah ada cukup untuk memulai ANN dan SVM.
