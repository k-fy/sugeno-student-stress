# Know Your Academic Stress 💌

Aplikasi web untuk memperkirakan tingkat stres akademik dari **jam tidur** dan **jam belajar** per hari, menggunakan logika fuzzy Sugeno. Proyek Soft Computing, Kelompok 6.

## Cara Kerja

- **Model**: Sugeno orde nol, AND = MIN, output berupa weighted average.
- **Himpunan fuzzy**: trapezoid, 3 himpunan per input, dioptimasi dengan Differential Evolution.
- **Output**: skor stres (z*) dengan kategori **Low**, **Moderate**, atau **High** (batas 32,5 dan 67,5).
- **Performa** (data uji, 600 baris): akurasi 0,9867 dan macro-F1 0,9837.

## Struktur File

| File | Fungsi |
|---|---|
| `app.py` | Tampilan Streamlit (form input, hasil, grafik, tab penjelasan) |
| `fuzzy_core.py` | Fungsi keanggotaan, rule base, dan inferensi fuzzy |
| `notebook.ipynb` | Notebook pengembangan model ([versi Kaggle](https://www.kaggle.com/code/putishas/sc-stress-levels-based-on-student-s-lifestyle)) |
| `requirements.txt` | Daftar library yang dibutuhkan |
| `README.md` | Dokumentasi proyek |

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Gunakan Streamlit versi terbaru, karena aplikasi memakai `st.iframe` dan `width="stretch"`.

## Cara Pakai

1. Isi **Sleep Hours** dan **Study Hours** (0–24, total tidak boleh lebih dari 24).
2. Klik **Submit**, halaman akan scroll otomatis ke hasil.
3. Lihat skor, kategori, proses fuzzifikasi, rule yang aktif, dan penjelasan model di tiap tab.

> Input di luar rentang data latih (5–10 jam) tetap dihitung, tetapi hasilnya merupakan ekstrapolasi.

## Tim

Kelompok 6 Soft Computing: Anggita Kirana Puspa, Puti Shasta Khafiyani, Varissa Nabila Kifli.
