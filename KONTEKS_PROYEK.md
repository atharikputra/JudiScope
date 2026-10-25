# JudiScope

https://github.com/atharikputra/JudiScope.git

Dokumen ini menyimpan konteks bersama proyek: apa yang dikerjakan, bagaimana eksperimen dibandingkan, dan bagaimana hasilnya disusun menjadi paper. Panduan teknis pemakaian dataset tersedia di [PANDUAN_DATASET_TIM.md](PANDUAN_DATASET_TIM.md).

## Identitas dan tujuan proyek

Nama proyek: **JudiScope**. Nama ini menggabungkan topik judi online dengan gagasan mengamati pola komentar dan mengevaluasi model secara terukur.

Proyek ini dikerjakan sebagai tugas besar pengganti UTS untuk klasifikasi data teks. Kelompok membandingkan tiga algoritma: Artificial Neural Network (ANN), Support Vector Machine (SVM), dan pendekatan Bayesian yang diimplementasikan sebagai Multinomial Naive Bayes.

Tujuannya adalah melatih model dari masing-masing algoritma, menguji model pada pembagian data yang sama, membahas perbedaan performa dan kesalahannya, kemudian menyusun hasil eksperimen menjadi paper dan presentasi. Algoritma boleh memakai implementasi library; pekerjaan utama kelompok adalah merancang eksperimen yang dapat direproduksi dan menjelaskan hasilnya.

Pertanyaan yang ingin dijawab:

1. Bagaimana performa ANN, SVM, dan Multinomial Naive Bayes dalam membedakan komentar promosi judi dari komentar nonjudi pada dataset ini?
2. Bagaimana perbedaan precision, recall, F1, accuracy, dan waktu latihan ketiga model?
3. Bagaimana penambahan komentar sintetis pada data latihan memengaruhi performa setiap algoritma?
4. Pola komentar seperti apa yang menyebabkan false positive dan false negative?

## Ruang lingkup klasifikasi

Masukan model adalah teks komentar YouTube yang sudah diproses. Keluaran model berupa prediksi kelas:

| Label | Makna kerja saat ini |
| --- | --- |
| `0` | Komentar nonjudi |
| `1` | Komentar promosi judi online |

Pemetaan tersebut mengikuti pemeriksaan contoh dan eksperimen yang sudah dibuat. Arti resmi label serta metode anotasinya masih perlu dikonfirmasi dari sumber dataset. Menyebut kata “judi” saja tidak otomatis berarti sebuah komentar merupakan promosi; kasus seperti kritik terhadap spam perlu dibahas dalam analisis kesalahan.

Ruang lingkup saat ini adalah eksperimen klasifikasi teks. Aplikasi moderasi, penghapusan komentar otomatis, dan integrasi langsung ke YouTube belum menjadi hasil yang dikerjakan dalam tahap ini.

## Dataset dan persiapannya

Sumber dataset awal yang disampaikan anggota kelompok adalah **[Judi Online di Kaggle](https://www.kaggle.com/datasets/yaemico/judionline)**, dengan identifier `yaemico/judionline`. Data digunakan sebagai data sekunder; kelompok mengerjakan EDA, cleaning, eksperimen model, dan augmentasi sintetis atas salinan lokalnya. Catatan atribusi tersedia di [dataset/README.md](dataset/README.md).

Judul dan URL halaman sumber sudah diperiksa, tetapi rincian deskripsi, versi rilis, lisensi, dan metode anotasi belum berhasil diverifikasi dari halaman tersebut. Angka jumlah record, video, dan channel di bawah berasal dari pemeriksaan file lokal, bukan klaim bahwa seluruh versi dataset Kaggle identik.

Dataset mentah berada di folder `dataset/`. Pemeriksaan awal menemukan 10.230 record komentar dari 32 video dan 18 channel. File sumber memiliki teks asli, teks hasil preprocessing lama, label, metadata, dan prediksi model lama.

Cleaning dilakukan dari kolom `komentar` asli agar prosesnya terdokumentasi. Tahapnya meliputi normalisasi Unicode NFKC, penghapusan karakter format tak terlihat, perapian karakter kontrol dan spasi, huruf kecil, serta penggantian URL berawalan http/https/www menjadi penanda `<url>`. Angka, simbol, dan emoji tetap dipertahankan. Stemming dan penghapusan stopword belum digunakan dalam baseline.

Setelah cleaning:

- 8.934 komentar asli unik dipertahankan.
- 1.294 record duplikat tambahan disisihkan berdasarkan teks hasil cleaning.
- 2 record dalam satu kelompok teks dengan label bertentangan dipisahkan untuk review, tanpa koreksi label otomatis.

Dataset bersih berada di `outputs/cleaning/komentar_cleaned.csv`. Gunakan `komentar_processed` sebagai input dan `label` sebagai target. ID, metadata, frekuensi duplikat, prediksi lama, dan penanda sintetis bukan fitur baseline.

Generator juga menghasilkan 1.066 komentar sintetis berlabel `1`, sehingga file gabungan berisi tepat 10.000 komentar: 8.934 asli dan 1.066 buatan. Komentar sintetis dibuat melalui kombinasi template dan variasi tulisan; jumlah tersebut tidak boleh dilaporkan sebagai 10.000 komentar YouTube asli. Generator dan hasilnya mencatat asal data.

## Rancangan eksperimen bersama

Ketiga algoritma mengikuti manifest `outputs/bayesian/original_split_manifest.csv`, dicocokkan lewat `source_row`. Jangan membuat pembagian baru untuk setiap algoritma.

| Bagian data asli | Jumlah | Fungsi |
| --- | ---: | --- |
| Train | 5.360 | Melatih kandidat model dan TF-IDF |
| Validation | 1.787 | Memilih parameter dan skenario |
| Test | 1.787 | Evaluasi akhir setelah pilihan dikunci |

Pembagian sekitar 60%/20%/20% menggunakan stratifikasi label dan seed `42`. Split dilakukan pada teks unik secara acak, bukan berdasarkan video atau kampanye.

Setiap algoritma menjalankan dua skenario:

| Skenario | Data latihan saat tuning | Validation dan test |
| --- | --- | --- |
| Tanpa sintetis | 5.360 komentar asli | Komentar asli dengan pembagian yang sama |
| Dengan sintetis | 5.360 asli + 1.066 sintetis | Komentar asli dengan pembagian yang sama |

Parameter dipilih berdasarkan F1 kelas `1` pada validation, dengan macro-F1 sebagai pemecah seri. Setelah pilihan dikunci, model dibuat ulang dan dilatih pada train + validation: 7.147 asli untuk skenario pertama atau 8.213 record termasuk sintetis untuk skenario kedua. Test tetap tidak digunakan untuk latihan atau pemilihan parameter.

Representasi teks awal adalah TF-IDF, dengan kandidat fitur kata unigram/bigram dan karakter 3–5 gram sesuai panduan tim. TF-IDF hanya di-fit pada data latihan masing-masing kandidat, lalu dipakai untuk mentransformasi validation. Saat refit, TF-IDF dipelajari ulang dari data development; test hanya ditransformasi.

## Algoritma yang dibandingkan

| Algoritma | Bentuk implementasi | Catatan eksperimen |
| --- | --- | --- |
| ANN | MLPClassifier, satu lapisan tersembunyi | 32/64 unit, ReLU, Adam, early stopping; lihat notebook 04 |
| SVM | LinearSVC | TF-IDF bersama, `C = 0.1, 1, 10`, L2 dan squared hinge; lihat notebook 05 |
| Bayesian | Multinomial Naive Bayes | Implementasi saat ini memakai MultinomialNB dengan tuning smoothing `alpha` |

Implementasi dan keluaran ketiga metode sudah tersedia: Bayesian oleh Atharik di notebook 03, ANN oleh Bagas di notebook 04, dan SVM oleh Aidan di notebook 05. Setiap metode menyimpan konfigurasi terpilih, skor validation/test, prediksi per record, serta versi library dalam folder keluarannya sendiri.

Perbandingan utama harus mempertahankan pembagian data, input, metrik, dan aturan pemilihan parameter. Bila representasi atau transformasi tambahan berbeda karena kebutuhan algoritma, jelaskan perbedaannya dalam metode. Catat juga anggaran tuning dan lingkungan eksekusi sebelum membandingkan waktu latihan.

## Hasil yang dibahas

Setiap anggota menyerahkan konfigurasi, skor validation, skor test, confusion matrix, waktu fit, prediksi per record test, dan contoh kesalahan. Metrik bersama adalah accuracy, precision kelas `1`, recall kelas `1`, F1 kelas `1`, dan macro-F1.

False positive berarti komentar berlabel nonjudi diprediksi sebagai promosi judi. False negative berarti komentar berlabel promosi judi diprediksi sebagai nonjudi. Keduanya dibahas bersama contoh teks dan kemungkinan masalah anotasi.

Eksperimen Bayesian awal sudah tersedia:

| Data latihan | Accuracy test | F1 kelas `1` pada test |
| --- | ---: | ---: |
| Tanpa sintetis | 97,87% | 96,79% |
| Dengan sintetis | 96,81% | 95,18% |

Hasil tersebut berasal dari 1.787 komentar test asli. Konfigurasi terbaik per skenario menggunakan TF-IDF `char_wb` 3–5 gram dan `alpha=0.1`. Skenario tanpa sintetis telah dipilih berdasarkan validation sebelum evaluasi test. Hasil ANN tersedia di `outputs/ann/`, sedangkan SVM tersedia di `outputs/svm/`. Gabungkan hasil berdasarkan `source_row` dan skenario saat mengerjakan pembandingan; versi lingkungan berbeda dan tidak boleh dianggap identik.

## Arah paper

Usulan judul paper:

**Perbandingan ANN, SVM, dan Multinomial Naive Bayes untuk Deteksi Komentar Promosi Judi Online di YouTube**

Judul bahasa Inggris bila diperlukan:

**A Comparative Study of ANN, SVM, and Multinomial Naive Bayes for Detecting Online Gambling Promotional Comments on YouTube**

Susunan pembahasan yang direncanakan:

1. **Pendahuluan:** masalah promosi judi pada komentar, tujuan klasifikasi, dan pertanyaan penelitian.
2. **Penelitian terkait:** pendekatan klasifikasi teks, spam/promosi judi, dan perbandingan model yang relevan. Referensi harus dicari, dibaca, dan diverifikasi sebelum dikutip.
3. **Metode:** asal dan anotasi dataset, EDA, cleaning, penanganan duplikat/konflik, generator sintetis, split, TF-IDF, parameter, dan evaluasi.
4. **Hasil dan pembahasan:** perbandingan tiga algoritma pada dua skenario, confusion matrix, kesalahan, waktu latihan, dan dampak augmentasi.
5. **Kesimpulan dan keterbatasan:** temuan yang didukung hasil, batas generalisasi, dan pekerjaan lanjutan.

Sesuai slide tugas, keluaran mencakup kode program, paper dengan ketentuan submit ke jurnal Sinta 3 atau 4, presentasi, serta deklarasi penggunaan AI dan plagiarisme. Makna persis batas persentase pada slide perlu dikonfirmasi kepada dosen. Target submit bukan klaim bahwa paper sudah diterima atau dipublikasikan.

Pembahasan dan kesimpulan memiliki bobot 40% pada slide tugas. Karena itu, paper perlu menjelaskan perbedaan hasil dan jenis kesalahan, bukan hanya menyajikan model dengan accuracy tertinggi.

## Berkas dan status pekerjaan

| Berkas | Peran / status |
| --- | --- |
| `01_eda_cleaning.ipynb` | EDA, cleaning, audit, dan ekspor data bersih; sudah dijalankan lewat Python |
| `02_dataset_generator.ipynb` | Generator sintetis, preview, verifikasi, dan ekspor; sudah dijalankan lewat Python |
| `03_bayesian_naive_bayes.ipynb` | Training, tuning, evaluasi, dan penyimpanan Naive Bayes; sudah dijalankan lewat Python |
| `04_ann.ipynb` | Training dan evaluasi MLP; hasil tersedia di `outputs/ann/` |
| `05_svm.ipynb` | Workflow SVM mandiri dalam JudiScope; seluruh sel telah dijalankan lewat IPython berurutan |
| `PANDUAN_DATASET_TIM.md` | Petunjuk teknis dataset dan protokol bersama untuk ANN/SVM |
| `outputs/cleaning/` | Data bersih, review konflik, audit, dan ringkasan |
| `outputs/augmentation/` | Tambahan sintetis, gabungan 10.000 record, dan ringkasan generator |
| `outputs/bayesian/` | Model Bayesian, manifest split, metrik, prediksi, dan grafik |
| `outputs/ann/` | Model ANN, metrik, prediksi, grafik, dan bobot proksi |
| `outputs/svm/` | Metrik, prediksi, koefisien, grafik, dan model SVM lokal yang diabaikan Git |
| Paper dan presentasi | Dikelola terpisah dari repository kode |

Keluaran ketiga metode sudah dapat digunakan untuk tabel perbandingan dan pembahasan kesalahan dalam paper atau presentasi. Notebook SVM membaca data dan manifest di repository ini tanpa membutuhkan repository SVM terpisah. Ringkasan setiap metode mencatat versi lingkungan dan batas hasilnya.

## Batas interpretasi

Kualitas anotasi sumber belum dipastikan. Deduplikasi hanya menghapus teks yang sama setelah cleaning; variasi template yang mirip dapat tetap tersebar pada train dan test. Generator dibuat berdasarkan gaya umum data yang sudah diperiksa, sehingga tambahan sintetis bukan sumber independen. Penanda nama demo atau pola template juga bisa menjadi jalan pintas bagi model.

Skor dari split acak dataset ini belum menunjukkan kemampuan pada video, kampanye, situs judi, atau periode waktu baru. Kesimpulan harus dibatasi pada dataset, pembagian, representasi teks, dan pengaturan eksperimen yang digunakan. Jangan mengubah parameter berkali-kali berdasarkan skor test yang sudah terlihat.
