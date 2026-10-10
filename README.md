# scikit-learn Cookbook, Third Edition

Reproduksi kode dan penjelasan teori dari buku **scikit-learn Cookbook, Third Edition: Over 80 recipes for machine learning in Python with scikit-learn** (John Sukup, Packt Publishing, 2025). Repositori ini mencakup **Chapter 1 sampai 5**.

Setiap chapter dikerjakan dalam satu notebook Jupyter yang berisi:
1. **Ringkasan chapter** di bagian awal.
2. **Reproduksi kode** tiap recipe, lengkap dengan output yang sudah dijalankan.
3. **Penjelasan teori** dan interpretasi hasil di bawah tiap sel kode.
4. **Kesimpulan chapter** di bagian akhir.



## Struktur repositori

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── Chapter-01/chapter01.ipynb   # Common Conventions and API Elements of scikit-learn
├── Chapter-02/chapter02.ipynb   # Pre-Model Workflow and Data Preprocessing
├── Chapter-03/chapter03.ipynb   # Dimensionality Reduction Techniques
├── Chapter-04/chapter04.ipynb   # Building Models with Distance Metrics and Nearest Neighbors
└── Chapter-05/chapter05.ipynb   # Linear Models and Regularization
```

## Cara menjalankan

```bash
git clone https://github.com/Son0fGuilliman/scikit-learn-cookbook
cd scikit-learn-cookbook

python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux:          source .venv/bin/activate

pip install -r requirements.txt
```

Buka notebook di VS Code (ekstensi Python dan Jupyter), pilih kernel `.venv`, lalu jalankan **Restart Kernel and Run All**. Jalankan sel dari atas ke bawah karena banyak recipe memakai variabel dari recipe sebelumnya.

Catatan lingkungan:
- Notebook diuji dengan Python 3.12, scikit-learn 1.8.0, pandas 3.0.2, NumPy 2.4.4, Matplotlib 3.10.8, dan seaborn 0.13.2. Buku memakai scikit-learn 1.5; hasil bagian yang direproduksi dari buku tetap sama, tetapi Chapter 1 memakai `sklearn.utils.get_tags` sehingga membutuhkan scikit-learn **1.6 atau lebih baru**.
- Latihan akhir pada notebook Chapter 2 memanggil `fetch_california_housing()` yang **mengunduh dataset** saat pertama kali dijalankan, sehingga perlu koneksi internet. Sel-sel latihan itu belum memiliki output sampai dijalankan di lingkunganmu.

## Konvensi dalam notebook

| Label sel | Arti |
|---|---|
| **[Dari buku]** | Kode yang direproduksi dari buku (kadang dengan perbaikan kecil yang dicatat di sel) |
| **[Tambahan]** | Kode atau pengecekan tambahan dari saya untuk membuktikan atau memperjelas konsep |
| **[Latihan]** | Solusi untuk latihan akhir recipe pada Chapter 2 (pipeline California Housing) |

## Ringkasan isi tiap chapter

### Chapter 1 — Common Conventions and API Elements of scikit-learn
Memperkenalkan konvensi API yang dipakai di seluruh library. Prinsip desainnya: konsistensi, kesederhanaan, modularitas, dan reusability. **Estimator** belajar dari data lewat `fit()` dan memprediksi lewat `predict()`; **transformer** mengubah data lewat `fit()` / `transform()`; `fit_predict()` dan `fit_transform()` adalah jalan pintas. Transformer hanya di-`fit` pada data latih untuk mencegah data leakage. Dibahas pula custom estimator (`BaseEstimator` dan mixin), `Pipeline` serta konteks MLOps, atribut dan method umum (`coef_`, `intercept_`, `score`), hyperparameter tuning (`GridSearchCV`, `RandomizedSearchCV`, `set_params` / `get_params`), metadata (estimator tags dan metadata routing), dan best practices.

### Chapter 2 — Pre-Model Workflow and Data Preprocessing
Kualitas data menentukan kualitas model (*garbage in, garbage out*). Isinya: penanganan **missing data** (`SimpleImputer`, `KNNImputer`, `IterativeImputer`), **scaling** (`StandardScaler`, `MinMaxScaler`, `Normalizer`), **encoding kategorikal** (`OneHotEncoder`, `LabelEncoder`, `ColumnTransformer`), **pipeline** yang menyatukan preprocessing dan model, serta **feature engineering** (`PolynomialFeatures`, `KBinsDiscretizer`, `RFE`, `SelectFromModel`). Ditutup dengan pipeline lengkap pada dataset California Housing.

### Chapter 3 — Dimensionality Reduction Techniques
Mengurangi jumlah fitur sambil mempertahankan informasi penting untuk menyederhanakan data, menghemat komputasi, mengurangi overfitting, dan memungkinkan visualisasi. **PCA** (unsupervised) memaksimalkan variansi; **LDA** (supervised) memaksimalkan keterpisahan kelas; **t-SNE** (non-linear) untuk visualisasi struktur lokal. Dibahas juga pedoman memilih teknik dan trade-off terhadap performa model (kompleksitas vs interpretabilitas, waktu vs akurasi, overfitting vs underfitting). Dataset: Wine dan Digits.

### Chapter 4 — Building Models with Distance Metrics and Nearest Neighbors
Model berbasis jarak, terutama **k-Nearest Neighbors (KNN)**. Dibahas metrik jarak (Euclidean, Manhattan, Minkowski), cara kerja KNN (voting dan pemilihan *k*), perbandingan metrik pada dataset sintetis, hyperparameter tuning dengan `GridSearchCV`, dan evaluasi (cross-validation, learning curve, confusion matrix, precision/recall/F1). Dataset: Iris dan data sintetis.

### Chapter 5 — Linear Models and Regularization
Dimulai dari regresi linear (OLS), lalu **Ridge** (penalti L2), **Lasso** (penalti L1, seleksi fitur), dan **ElasticNet** (gabungan, diatur oleh `alpha` dan `l1_ratio`). Teori regularisasi dan bias-variance trade-off dijelaskan, lalu regresi linear diperluas ke hubungan non-linear dengan **polynomial regression** dan **spline** (`SplineTransformer`). Dataset: data regresi sintetis.

## Catatan temuan saat reproduksi

Beberapa kode atau pernyataan di buku perlu penyesuaian agar dapat berjalan atau akurat. Semuanya dicatat di notebook terkait. Contoh yang paling penting:

- **Chapter 3:** panah pada plot PCA di buku menggambar bobot dua fitur pertama, bukan arah komponen; `load_digits` bukan MNIST.
- **Chapter 4:** import `cross_val_score` dan `classification_report` tidak ditampilkan di buku; `euclidean` dan `minkowski` (p=2 default) secara matematis identik; `make_circles` diberi `random_state` agar reproducible.
- **Chapter 5:** dataset sintetis buku menimpa 7 dari 10 fitur informatif sehingga R² model rendah (≈0,06 pada seed yang dipakai); keunggulan ElasticNet pada tabel buku terutama karena skala `alpha` (setara `Ridge(alpha=400)`); kode tabel metrik ElasticNet dapat menghasilkan `TypeError` pada pandas 3.x karena memformat angka menjadi string sebelum di-sort.

## Referensi

- John Sukup, *scikit-learn Cookbook, Third Edition*, Packt Publishing, 2025.
- Kode contoh resmi dari penerbit: <https://github.com/PacktPublishing/scikit-learn-Cookbook-Third-Edition>
- Dokumentasi scikit-learn: <https://scikit-learn.org/stable/>
