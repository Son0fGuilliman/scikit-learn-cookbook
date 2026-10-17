# scikit-learn Cookbook, Third Edition (Packt) - Code & Concept Reproductions

Repositori ini berisi dokumentasi, penjelasan teori, dan reproduksi kode lengkap dalam format Jupyter Notebook dari buku **"scikit-learn Cookbook, Third Edition: Over 80 recipes for machine learning in Python with scikit-learn"** (oleh John Sukup, terbitan Packt Publishing, 2025).

Repositori ini disusun untuk memenuhi tugas akademis serta menjadi referensi praktis penerapan scikit-learn: dari konvensi API, preprocessing, reduksi dimensi, hingga model klasifikasi dan regresi. Cakupan: **Chapter 1 sampai 8**.

---

## 📁 Struktur Repositori

```
.
├── ScikitLearnCookbookChapter1.ipynb    # Chapter 1: Common Conventions and API Elements of scikit-learn
├── ScikitLearnCookbookChapter2.ipynb    # Chapter 2: Pre-Model Workflow and Data Preprocessing
├── ScikitLearnCookbookChapter3.ipynb    # Chapter 3: Dimensionality Reduction Techniques
├── ScikitLearnCookbookChapter4.ipynb    # Chapter 4: Building Models with Distance Metrics and Nearest Neighbors
├── ScikitLearnCookbookChapter5.ipynb    # Chapter 5: Linear Models and Regularization
├── ScikitLearnCookbookChapter6.ipynb    # Chapter 6: Advanced Logistic Regression and Extensions
├── ScikitLearnCookbookChapter7.ipynb    # Chapter 7: Support Vector Machines and Kernel Methods
├── ScikitLearnCookbookChapter8.ipynb    # Chapter 8: Tree-Based Algorithms and Ensemble Methods
├── requirements.txt
├── .gitignore
└── README.md                             # Rangkuman proyek & ringkasan materi
```

> Tidak ada folder `data/` karena seluruh dataset berasal dari `sklearn.datasets` atau dibangkitkan secara sintetis. Satu-satunya pengecualian adalah *California Housing* di latihan akhir Chapter 2, yang diunduh otomatis oleh `fetch_california_housing()` (perlu internet).

---

# 📚 Ringkasan Materi Per Chapter

> Catatan ringkas mengenai konsep-konsep utama pada tiap chapter. Penjelasan lengkap, rumus, dan interpretasi hasil ada di notebook masing-masing.

## Daftar Isi

- [Chapter 1: Common Conventions and API Elements of scikit-learn](#chapter-1-common-conventions-and-api-elements-of-scikit-learn)
- [Chapter 2: Pre-Model Workflow and Data Preprocessing](#chapter-2-pre-model-workflow-and-data-preprocessing)
- [Chapter 3: Dimensionality Reduction Techniques](#chapter-3-dimensionality-reduction-techniques)
- [Chapter 4: Building Models with Distance Metrics and Nearest Neighbors](#chapter-4-building-models-with-distance-metrics-and-nearest-neighbors)
- [Chapter 5: Linear Models and Regularization](#chapter-5-linear-models-and-regularization)
- [Chapter 6: Advanced Logistic Regression and Extensions](#chapter-6-advanced-logistic-regression-and-extensions)
- [Chapter 7: Support Vector Machines and Kernel Methods](#chapter-7-support-vector-machines-and-kernel-methods)
- [Chapter 8: Tree-Based Algorithms and Ensemble Methods](#chapter-8-tree-based-algorithms-and-ensemble-methods)
- [Catatan Reproduksi](#catatan-reproduksi)
- [Cara Menjalankan](#cara-menjalankan)

---

## Chapter 1: Common Conventions and API Elements of scikit-learn

### Filosofi Desain & API Seragam
- **Prinsip desain**: konsistensi, kesederhanaan, modularitas, dan *reusability*.
- **Estimator**: `fit()` untuk belajar dan `predict()` untuk memprediksi. **Transformer**: `fit()` / `transform()`. Jalan pintas: `fit_predict()` dan `fit_transform()`.
- **Mencegah data leakage**: transformer hanya di-`fit` pada data latih, lalu `transform` pada data uji.

### Custom Estimator, Pipeline & Tuning
- **Custom estimator** dibuat dengan `BaseEstimator` dan mixin (`TransformerMixin`, dst.).
- **Pipeline** menyatukan preprocessing dan model; relevan dengan praktik MLOps.
- **Atribut & method umum**: `coef_`, `intercept_`, `score()`; tuning dengan `GridSearchCV` / `RandomizedSearchCV`, `set_params()` / `get_params()`.
- **Metadata**: *estimator tags* dan *metadata routing*; ditutup dengan *best practices* penggunaan API.

---

## Chapter 2: Pre-Model Workflow and Data Preprocessing

### Missing Data
- **Imputasi**: `SimpleImputer` (mean/median/modus), `KNNImputer` (tetangga terdekat), `IterativeImputer` (regresi antarfitur).

### Scaling & Encoding
- **Scaling**: `StandardScaler` (mean 0, std 1), `MinMaxScaler` (rentang 0-1), `Normalizer` (norma tiap baris = 1).
- **Encoding kategorikal**: `OneHotEncoder` (nominal), `OrdinalEncoder` (ordinal); `LabelEncoder` sebenarnya untuk target. `ColumnTransformer` untuk preprocessing berbeda per kolom.

### Pipeline & Feature Engineering
- **Pipeline** dengan prinsip split dulu baru transformasi untuk menghindari *data leakage*.
- **Feature engineering**: `PolynomialFeatures`, `KBinsDiscretizer`; **seleksi fitur**: `RFE`, `SelectFromModel`.

---

## Chapter 3: Dimensionality Reduction Techniques

### Feature Selection vs Feature Extraction
- Tujuan reduksi dimensi: menyederhanakan data, menghemat komputasi, mengurangi overfitting, dan memudahkan visualisasi.

### PCA, LDA, dan t-SNE
- **PCA** (*unsupervised*): komponen ortogonal yang memaksimalkan variansi; data harus distandarisasi.
- **LDA** (*supervised*): memaksimalkan keterpisahan kelas; komponen maksimum $\min(n_{fitur}, n_{kelas}-1)$.
- **t-SNE** (*non-linear*): visualisasi struktur lokal dan klaster; stokastik dan tidak punya `transform()` untuk data baru.

### Memilih Teknik & Dampak pada Model
- **Pedoman pemilihan** berdasarkan tujuan, ketersediaan label, ukuran data, dan interpretabilitas.
- **Trade-off**: kompleksitas vs interpretabilitas, waktu vs akurasi, overfitting vs underfitting.

---

## Chapter 4: Building Models with Distance Metrics and Nearest Neighbors

### Metrik Jarak
- **Euclidean**, **Manhattan**, dan **Minkowski** (parameter $p$; $p=1$ Manhattan, $p=2$ Euclidean).

### k-Nearest Neighbors (KNN)
- Klasifikasi lewat *voting* $k$ tetangga terdekat (regresi: rata-rata); *lazy learner*.
- **Scaling** penting karena jarak dipengaruhi skala fitur.

### Tuning & Evaluasi
- **Tuning**: `GridSearchCV` / `RandomizedSearchCV` untuk `n_neighbors`, `weights`, `metric`.
- **Evaluasi**: cross-validation, *learning curve*, *confusion matrix*, precision/recall/F1.

---

## Chapter 5: Linear Models and Regularization

### Regresi Linear
- **OLS** meminimalkan jumlah kuadrat galat; evaluasi dengan **MSE** dan **R²**.

### Ridge, Lasso, ElasticNet
- **Ridge (L2)**: menyusutkan koefisien tanpa menolkan. **Lasso (L1)**: dapat menolkan koefisien (seleksi fitur). **ElasticNet**: gabungan lewat `alpha` dan `l1_ratio`.
- **Regularisasi** menukar sedikit bias dengan penurunan variansi (*bias-variance trade-off*).

### Hubungan Non-Linear
- **Polynomial regression** dan **spline** (`SplineTransformer`) memperluas model linear; polinomial derajat tinggi berisiko overfitting.

---

## Chapter 6: Advanced Logistic Regression and Extensions

### Regresi Logistik
- Probabilitas lewat **sigmoid**: $P(Y=1\mid X)=\frac{1}{1+e^{-z}}$; model linear pada **log-odds**; diestimasi dengan **MLE**.

### Multiclass
- **One-vs-Rest**: satu classifier biner per kelas. **Multinomial (softmax)**: satu model gabungan dengan probabilitas berjumlah 1.

### Regularisasi
- **L2/Ridge** menyusutkan koefisien, **L1/Lasso** menolkan koefisien; parameter **`C`** = kebalikan kekuatan regularisasi.

### Multilabel & Evaluasi
- **Multilabel**: tiap sampel dapat memiliki beberapa label; *binary relevance*, `MultiOutputClassifier`, matriks *co-occurrence*; metrik micro/macro/weighted/samples.
- **Metrik**: accuracy, precision, recall, F1, kurva **ROC** dan **AUC**.

---

## Chapter 7: Support Vector Machines and Kernel Methods

### SVM & Margin
- **Hyperplane margin maksimum**; *support vectors*; **soft margin** lewat parameter `C`. Tersedia untuk klasifikasi (`SVC`) dan regresi (`SVR`).

### Fungsi Kernel
- **Kernel trick**: hasil kali dalam di ruang berdimensi tinggi tanpa pemetaan eksplisit. Kernel **linear**, **polynomial**, **RBF** (parameter `degree`, `gamma`).

### Tuning, Dimensi Tinggi & Evaluasi
- **Tuning** `C`, `kernel`, dan parameter kernel dengan grid search + cross-validation.
- **Data berdimensi tinggi**: kernel linear dan regularisasi; visualisasi dengan PCA.
- **Evaluasi**: classification report dan ROC-AUC (`probability=True` memakai kalibrasi internal).

---

## Chapter 8: Tree-Based Algorithms and Ensemble Methods

### Decision Tree
- Pembelahan rekursif dengan **Gini impurity** atau **entropy**; dikendalikan oleh `max_depth`, `min_samples_split`, `min_samples_leaf`, dkk.

### Ensemble
- **Random forest (bagging)**: pohon pada sampel bootstrap + fitur acak; menurunkan variansi; *feature importance*.
- **Gradient boosting**: pohon berurutan yang memperbaiki residual; menurunkan bias; `learning_rate`, `n_estimators`.
- **Stacking**: model dasar beragam digabung oleh meta-model.

### Tuning & Perbandingan
- **Tuning** dengan `GridSearchCV` / `RandomizedSearchCV`; perbandingan bagging vs boosting vs stacking.

---

## Catatan Reproduksi

Beberapa kode atau pernyataan di buku perlu disesuaikan agar berjalan di scikit-learn terbaru atau agar akurat. Semuanya dicatat di notebook terkait. Contoh yang paling penting:

- **Chapter 3**: panah pada plot PCA di buku menggambar bobot dua fitur pertama, bukan arah komponen; `load_digits` bukan MNIST.
- **Chapter 4**: import `cross_val_score` dan `classification_report` tidak ditampilkan; `euclidean` dan `minkowski` (p=2) identik.
- **Chapter 5**: dataset sintetis buku menimpa 7 dari 10 fitur informatif sehingga R² rendah; keunggulan ElasticNet terutama karena skala `alpha`.
- **Chapter 6**: `multi_class` dihapus dan `liblinear` tidak lagi mendukung multiclass di scikit-learn 1.7+ (diganti `OneVsRestClassifier` dan multinomial default); ada *SyntaxError* pada fungsi *co-occurrence* di buku (koma hilang).
- **Chapter 7**: recipe evaluasi mencetak `accuracy` tanpa menghitungnya; hyperplane bukan "garis non-linear"; kontur "decision boundary" yang ditampilkan berasal dari `SVR`.
- **Chapter 8**: nilai maksimum Gini adalah $1-1/K$ (bukan 1); `RandomizedSearchCV` tidak adaptif; heatmap `pivot_table` merata-ratakan `learning_rate`.
- **Selisih akurasi kecil pada data uji Iris (45 sampel)** pada Chapter 4, 6, 7, dan 8 hanya setara 1-2 sampel, sehingga tidak cukup untuk menyimpulkan satu metode lebih baik.

Bagian *Hands-on exercises* / *Practical exercises* dari buku tidak dimasukkan pada Chapter 3 sampai 8. Chapter 1 dan 2 dikerjakan lebih dulu dengan format berbeda; Chapter 2 masih memuat latihan akhir pipeline California Housing (butuh internet, sel latihan belum memiliki output sampai dijalankan).

---

## Cara Menjalankan

```bash
git clone https://github.com/<username>/<nama-repo>.git
cd <nama-repo>

python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux:          source .venv/bin/activate

pip install -r requirements.txt
```

Buka notebook di VS Code (ekstensi Python dan Jupyter), pilih kernel `.venv`, lalu jalankan **Restart Kernel and Run All**. Jalankan sel dari atas ke bawah karena banyak recipe memakai variabel dari recipe sebelumnya.

Notebook diuji dengan Python 3.12, scikit-learn 1.8.0, pandas 3.0.2, NumPy 2.4.4, Matplotlib 3.10.8, dan seaborn 0.13.2. Buku memakai scikit-learn 1.5; Chapter 1 memakai `sklearn.utils.get_tags` sehingga membutuhkan scikit-learn **1.6 atau lebih baru**.
