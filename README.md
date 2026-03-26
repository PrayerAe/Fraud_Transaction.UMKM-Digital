# Deteksi Fraud pada Transaksi Pembayaran Digital UMKM

Analisis komparatif kinerja algoritma Machine Learning untuk mendeteksi transaksi fraud pada sistem pembayaran digital UMKM, dilengkapi dengan Feature Engineering mendalam dan Explainable AI (XAI) menggunakan SHAP.

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Dataset](#dataset)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Alur Kerja Pipeline](#alur-kerja-pipeline)
  - [Tahap 1: Pemuatan Data](#tahap-1-pemuatan-data)
  - [Tahap 2: Feature Engineering](#tahap-2-feature-engineering-11-kolom--42-fitur)
  - [Tahap 3: Penanganan Ketidakseimbangan Kelas](#tahap-3-penanganan-ketidakseimbangan-kelas-smote)
  - [Tahap 4: Pelatihan Model](#tahap-4-pelatihan-model-xgboost)
  - [Tahap 5: Explainable AI (SHAP)](#tahap-5-explainable-ai-xai-dengan-shap)
- [Struktur File](#struktur-file)
- [Hasil Analisis](#hasil-analisis)
- [Cara Menjalankan](#cara-menjalankan)
- [Dependensi](#dependensi)

## Latar Belakang

Pertumbuhan transaksi digital pada sektor UMKM membawa risiko fraud yang semakin kompleks. Sistem deteksi fraud konvensional berbasis aturan (rule-based) memiliki keterbatasan dalam mengenali pola fraud yang terus berevolusi. Penelitian ini mengimplementasikan pendekatan Machine Learning dengan penekanan pada:

1. **Feature Engineering yang mendalam** - Mengekstrak 42 fitur bermakna dari 11 kolom data mentah
2. **Explainable AI (XAI)** - Memberikan transparansi mengapa suatu transaksi diklasifikasikan sebagai fraud, memenuhi kebutuhan regulasi dan kepercayaan stakeholder

## Dataset

**PaySim** - Dataset sintetis yang disimulasikan berdasarkan pola transaksi mobile money nyata dari sebuah perusahaan fintech di Afrika.

| Properti | Detail |
|----------|--------|
| Sumber | [Kaggle - PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| File | `PS_20174392719_1491204439457_log.csv` (~493 MB) |
| Jumlah Transaksi | 6.362.620 |
| Periode Simulasi | 30 hari (720 step, 1 step = 1 jam) |
| Rasio Fraud | ~0,13% (8.213 transaksi fraud) |
| Jenis Transaksi | PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT |

### Kolom Data Asli

| Kolom | Deskripsi |
|-------|-----------|
| `step` | Unit waktu simulasi (1 step = 1 jam) |
| `type` | Jenis transaksi |
| `amount` | Nominal transaksi |
| `nameOrig` | ID akun pengirim |
| `oldbalanceOrg` | Saldo pengirim sebelum transaksi |
| `newbalanceOrig` | Saldo pengirim sesudah transaksi |
| `nameDest` | ID akun penerima |
| `oldbalanceDest` | Saldo penerima sebelum transaksi |
| `newbalanceDest` | Saldo penerima sesudah transaksi |
| `isFraud` | Label ground truth (0 = normal, 1 = fraud) |
| `isFlaggedFraud` | Flag dari sistem deteksi lama |

### Mekanisme Fraud dalam PaySim

Label fraud **tidak diinjeksi secara manual**, melainkan dihasilkan oleh simulator PaySim yang memodelkan skenario fraud dunia nyata:

```
1. Fraudster mengambil alih akun korban (account takeover)
2. Seluruh saldo korban di-TRANSFER ke akun perantara (mule account)
3. Dana di-CASH_OUT dari akun perantara sehingga uang hilang dari sistem
```

Pola ini menghasilkan anomali yang dapat dideteksi melalui feature engineering, seperti saldo yang terkuras habis (`newbalanceOrig = 0`) dan ketidakcocokan persamaan saldo (`oldBalance - amount != newBalance`).

## Arsitektur Sistem

```
                    +---------------------------+
                    |   PaySim Dataset (CSV)    |
                    |   6,3 juta transaksi      |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |   Sampling (100.000)      |
                    |   untuk efisiensi         |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |   Feature Engineering     |
                    |   11 kolom -> 42 fitur    |
                    |   8 kategori fitur        |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |   Train/Test Split        |
                    |   70% train / 30% test    |
                    |   Stratified sampling     |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |   SMOTE Oversampling      |
                    |   Menyeimbangkan kelas    |
                    |   fraud vs normal         |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |   StandardScaler          |
                    |   Normalisasi fitur       |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |   Training XGBoost        |
                    |   (GPU accelerated)       |
                    +-------------+-------------+
                                  |
                        +---------+---------+
                        |                   |
                        v                   v
              +----------------+   +------------------+
              |   Evaluasi     |   |   SHAP Analysis  |
              |   Precision    |   |   - Summary Plot |
              |   Recall       |   |   - Bar Plot     |
              |   F1-Score     |   |   - Waterfall    |
              |   AUC-ROC      |   |   - Dependence   |
              +----------------+   +------------------+
                                            |
                                            v
                                   +------------------+
                                   |   XAI Report     |
                                   |   (Interpretasi) |
                                   +------------------+
```

## Alur Kerja Pipeline

### Tahap 1: Pemuatan Data

Dataset PaySim dimuat dari file CSV. Untuk efisiensi pengembangan, digunakan sampling 100.000 transaksi dari total 6,3 juta dengan `random_state=42` untuk reprodusibilitas.

```python
pipeline.load_and_prepare_data(
    data_path='PS_20174392719_1491204439457_log.csv',
    sample_size=100000
)
```

### Tahap 2: Feature Engineering (11 Kolom -> 42 Fitur)

Inti dari sistem ini adalah proses feature engineering yang mengekstrak informasi tersembunyi dari data mentah. Fitur-fitur dikelompokkan dalam 8 kategori:

#### Kategori 1: Balance Error Features (4 fitur)

Dalam transaksi normal, persamaan berikut **harus** berlaku:
- Pengirim: `oldbalanceOrg - amount = newbalanceOrig`
- Penerima: `oldbalanceDest + amount = newbalanceDest`

Jika persamaan ini tidak seimbang, ada indikasi manipulasi. Fitur ini terbukti **paling penting** dalam deteksi fraud (SHAP value tertinggi).

| Fitur | Formula | Hipotesis |
|-------|---------|-----------|
| `errorBalanceOrig` | `oldbalanceOrg - amount - newbalanceOrig` | Nilai != 0 mengindikasikan anomali |
| `errorBalanceDest` | `oldbalanceDest + amount - newbalanceDest` | Nilai != 0 mengindikasikan anomali |
| `totalBalanceError` | `abs(errorOrig) + abs(errorDest)` | Nilai tinggi = probabilitas fraud tinggi |
| `hasBalanceError` | `1 jika abs(error) > 0.01` | Flag biner keberadaan error |

#### Kategori 2: Temporal Features (4 fitur)

Mengekstrak pola waktu karena fraud cenderung terjadi pada jam-jam tertentu.

| Fitur | Formula | Hipotesis |
|-------|---------|-----------|
| `hour_of_day` | `step % 24` | Fraud dominan di malam hari |
| `is_night` | `1 jika jam 00:00-06:00` | Transaksi malam lebih berisiko |
| `is_peak_hour` | `1 jika jam 09:00-17:00` | Transaksi normal dominan di jam kerja |
| `day_of_simulation` | `step // 24` | Pola fraud berubah seiring waktu |

#### Kategori 3: Customer Aggregation Features (6 fitur)

Menangkap perilaku historis pelanggan sebagai konteks.

| Fitur | Deskripsi |
|-------|-----------|
| `trans_count` | Total jumlah transaksi per customer |
| `avg_amount` | Rata-rata nominal transaksi |
| `std_amount` | Variabilitas nominal (pola tidak konsisten = curiga) |
| `total_amount` | Total kumulatif transaksi |
| `min_amount` / `max_amount` | Rentang nominal transaksi |
| `account_age_steps` | Lama aktif akun (akun baru = lebih berisiko) |

#### Kategori 4: Transaction Pattern Features (4 fitur)

Mendeteksi pola transaksi yang menyimpang dari kebiasaan pelanggan.

| Fitur | Formula | Hipotesis |
|-------|---------|-----------|
| `deviation_from_avg` | `abs(amount - avg_amount)` | Deviasi besar = anomali |
| `amount_to_avg_ratio` | `amount / (avg_amount + 1)` | Rasio >> 1 = mencurigakan |
| `is_amount_outlier` | Z-score > 2 standar deviasi | Outlier berkorelasi kuat dengan fraud |
| `amount_to_total_ratio` | `amount / total_amount` | Satu transaksi besar = red flag |

#### Kategori 5: Balance Ratio Features (6 fitur)

Rasio saldo memberikan konteks kondisi keuangan akun.

| Fitur | Deskripsi |
|-------|-----------|
| `orig_balance_ratio` | Rasio saldo baru/lama pengirim (0 = terkuras) |
| `dest_balance_ratio` | Rasio saldo baru/lama penerima |
| `amount_to_orig_balance` | Proporsi transaksi terhadap saldo (rasio mendekati 1 = menguras akun) |
| `orig_zero_balance` | Flag saldo awal pengirim = 0 |
| `dest_zero_balance` | Flag saldo awal penerima = 0 |
| `orig_depleted` | Flag saldo akhir pengirim = 0 (akun terkuras habis) |

#### Kategori 6: Transaction Type Encoding (5 fitur)

One-hot encoding untuk jenis transaksi. Fraud hanya terjadi pada tipe **TRANSFER** dan **CASH_OUT**.

#### Kategori 7: Risk Indicator Features (3 fitur)

Indikator risiko komposit yang menggabungkan beberapa sinyal.

| Fitur | Deskripsi |
|-------|-----------|
| `is_high_value` | Transaksi di atas persentil ke-95 |
| `is_new_customer` | Customer dengan <= 2 transaksi |
| `risk_score` | Skor risiko komposit (weighted sum dari beberapa flag risiko) |

Formula risk score:
```
risk_score = hasBalanceError * 3
           + is_amount_outlier * 2
           + is_high_value * 1
           + is_new_customer * 1
           + is_night * 1
           + orig_depleted * 2
```

#### Kategori 8: Velocity Features (3 fitur)

Mendeteksi urutan transaksi yang terlalu cepat (indikasi otomatisasi/bot).

| Fitur | Deskripsi |
|-------|-----------|
| `trans_per_day` | Frekuensi transaksi per hari |
| `time_since_last` | Interval waktu sejak transaksi terakhir |
| `is_rapid_transaction` | Flag jika interval < 5 step (sangat cepat) |

### Tahap 3: Penanganan Ketidakseimbangan Kelas (SMOTE)

Dataset memiliki ketidakseimbangan ekstrem: hanya ~0,13% transaksi yang fraud. Tanpa penanganan, model akan bias memprediksi semua transaksi sebagai normal.

**SMOTE (Synthetic Minority Over-sampling Technique)** digunakan untuk membuat sampel sintetis kelas minoritas (fraud) pada data training:

```
Sebelum SMOTE:  Normal=99,87%  |  Fraud=0,13%
Sesudah SMOTE:  Normal=50%     |  Fraud=50%
```

SMOTE hanya diterapkan pada **data training**, bukan data test, untuk menghindari data leakage.

### Tahap 4: Pelatihan Model (XGBoost)

Model utama yang digunakan adalah **XGBoost** (eXtreme Gradient Boosting) dengan konfigurasi:

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| `max_depth` | 6 | Kedalaman tree yang cukup tanpa overfitting |
| `learning_rate` | 0.1 | Learning rate standar |
| `n_estimators` | 100 | Jumlah tree |
| `scale_pos_weight` | auto | Bobot tambahan untuk kelas fraud |
| `tree_method` | `gpu_hist` | Akselerasi GPU (jika tersedia) |

Evaluasi dilakukan dengan metrik: **Precision, Recall, F1-Score, dan AUC-ROC**.

### Tahap 5: Explainable AI (XAI) dengan SHAP

Setelah model dilatih, **SHAP (SHapley Additive exPlanations)** digunakan untuk menjelaskan mengapa model membuat keputusan tertentu. SHAP berasal dari konsep **Shapley Value** dalam teori permainan, yang menghitung kontribusi adil setiap "pemain" (fitur) terhadap hasil akhir (prediksi).

#### Cara Kerja SHAP

Untuk setiap prediksi, SHAP menghitung kontribusi setiap fitur:

```
Base value (rata-rata prediksi):     0.13

  + errorBalanceOrig tinggi          +0.45  (saldo tidak cocok)
  + orig_depleted = 1                +0.25  (akun terkuras)
  + amount besar                     +0.08  (nominal tinggi)
  + is_night = 1                     +0.04  (transaksi malam)
  ─────────────────────────────────────────
  = Prediksi akhir:                   0.95  (FRAUD)
```

#### Visualisasi SHAP yang Dihasilkan

| Visualisasi | File Output | Fungsi |
|-------------|-------------|--------|
| **Summary Plot** | `xai_summary_XGBoost.png` | Ringkasan dampak semua fitur terhadap prediksi |
| **Bar Plot** | `xai_bar_XGBoost.png` | Ranking fitur berdasarkan rata-rata SHAP value |
| **Waterfall Plot** | `xai_waterfall_XGBoost_instance0.png` | Penjelasan detail satu transaksi spesifik |
| **Dependence Plot** | `xai_dependence_XGBoost_*.png` | Hubungan antara nilai fitur dan dampaknya terhadap prediksi |

#### Hasil SHAP - Top 5 Fitur Terpenting

| Rank | Fitur | Mean SHAP | Interpretasi |
|------|-------|-----------|-------------|
| 1 | `errorBalanceOrig` | 0.4077 | Ketidakcocokan saldo pengirim (indikator terkuat) |
| 2 | `orig_depleted` | 0.2020 | Akun pengirim terkuras habis |
| 3 | `amount` | 0.0011 | Nominal transaksi |
| 4 | `errorBalanceDest` | 0.0005 | Ketidakcocokan saldo penerima |
| 5 | `orig_balance_ratio` | 0.0004 | Rasio perubahan saldo pengirim |

## Struktur File

```
FRAUD_TRANSACTION/
|
|-- fraud.py                    # Pipeline utama (versi GPU)
|-- fraud2.py                   # Pipeline versi CPU dengan debugging tambahan
|
|-- feature_engineering_catalog.csv   # Katalog lengkap 42 fitur yang dibuat
|-- feature_importance_XGBoost.csv    # Ranking fitur berdasarkan SHAP
|-- xai_report_XGBoost.txt           # Laporan XAI lengkap
|
|-- xai_summary_XGBoost.png                        # SHAP summary plot
|-- xai_bar_XGBoost.png                            # SHAP bar plot
|-- xai_waterfall_XGBoost_instance0.png            # SHAP waterfall plot
|-- xai_dependence_XGBoost_amount.png              # SHAP dependence: amount
|-- xai_dependence_XGBoost_errorBalanceOrig.png    # SHAP dependence: errorBalanceOrig
|-- xai_dependence_XGBoost_orig_depleted.png       # SHAP dependence: orig_depleted
|
|-- PS_20174392719_1491204439457_log.csv  # Dataset PaySim (tidak di-track git)
|-- README.md
|-- .gitignore
```

## Hasil Analisis

### Temuan Utama

1. **Balance Error adalah indikator fraud terkuat** - Fitur `errorBalanceOrig` memiliki SHAP value 0.4077, jauh melampaui fitur lainnya. Ini berarti ketidakcocokan antara saldo yang diharapkan dan saldo aktual adalah sinyal fraud paling kuat.

2. **Pola "account draining" sangat khas** - Fitur `orig_depleted` (SHAP: 0.2020) menunjukkan bahwa akun yang terkuras hingga saldo 0 berkorelasi sangat kuat dengan fraud.

3. **Fraud hanya terjadi pada TRANSFER dan CASH_OUT** - Sesuai dengan skenario fraud dalam PaySim di mana fraudster mentransfer dana ke akun perantara lalu mencairkannya.

4. **Top 3 fitur mendominasi prediksi** - `errorBalanceOrig`, `orig_depleted`, dan `amount` menyumbang mayoritas keputusan model, memberikan interpretasi yang jelas dan actionable.

## Cara Menjalankan

### Prasyarat

1. Download dataset PaySim dari [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
2. Letakkan file `PS_20174392719_1491204439457_log.csv` di direktori root project
3. Install dependensi yang diperlukan

### Menjalankan

```bash
# Versi GPU (membutuhkan CUDA)
python fraud.py

# Versi CPU
python fraud2.py
```

## Dependensi

| Library | Fungsi |
|---------|--------|
| `pandas`, `numpy` | Manipulasi data |
| `scikit-learn` | Preprocessing, evaluasi, model ML |
| `imbalanced-learn` | SMOTE dan teknik sampling lainnya |
| `xgboost` | Model XGBoost |
| `lightgbm` | Model LightGBM |
| `torch` | Neural network (PyTorch) |
| `shap` | Explainable AI |
| `matplotlib`, `seaborn` | Visualisasi |

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm torch shap matplotlib seaborn
```
