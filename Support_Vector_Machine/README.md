# ⚔️ Support Vector Machine (SVM) Classification — Oranges vs Grapefruit

Panduan praktikum membangun model klasifikasi menggunakan algoritma **Support Vector Machine (SVM)** dengan dataset Oranges vs Grapefruit.

---

## 📦 Dataset

- **Sumber:** [Kaggle – Oranges vs Grapefruit](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)
- **Jumlah data:** 10.000 baris
- **Target:** Kolom `name` (orange, grapefruit)
- **Fitur:** diameter, weight, red, green, blue

---

## ⚙️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 🚀 Langkah-Langkah

### 1. Import Library

> Ekstrak perkakas library yang dibutuhkan dalam pembentukan klasifikasi menggunakan Support Vector Machines.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
```

---

### 2. Load Data

> Ambil dan baca set data menggunakan framework pandas.

```python
df = pd.read_csv('../citrus.csv')
print("Total sampel:", len(df))
```

---

### 3. Preprocessing

> Gunakan label encoding dari sklearn untuk memformat column kategorikal `name` yang mengandung nilai teks, ke dalam tipe bilangan bulat yang terbaca algoritma matriks. 

```python
le = LabelEncoder()
df['name'] = le.fit_transform(df['name'])
```

---

### 4. Split Data & Scaling

> SVM merupakan algoritma yang rentan dan tersensitisasi dengan besaran variabel, jadinya standar skalasi adalah **kewajiban**. Potong menjadi blok latih dan blok uji.

```python
X = df.drop(columns=['name']).values
y = df['name'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
```

---

### 5. Training Model

> Lakukan inisialisasi pada Classifier SVC. Kita dapat menggunakan fungsi kemiringan Kernel (seperti Linear, rbf, atau poly). Default yang digunakan adalah 'rbf'.

```python
classifier = SVC(kernel='rbf', random_state=42)
classifier.fit(X_train, y_train)
```

---

### 6. Prediksi

> Simulasikan prediksi terhadap variable X hasil pengujian.

```python
y_pred = classifier.predict(X_test)
```

---

### 7. Evaluasi Model

> Uji tingkat akurasi untuk melihat seberapa kuat algoritma menemukan hyperplane (garis pisah) terbaik agar kelas jeruk dan anggur tertata dengan sempurna.

#### 📌 Accuracy & Classification Report

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
```

#### 📌 Confusion Matrix
> Memplot heat map matrix membuktikan persebaran predikasi yang salah dan yang tepat pada kedua label buah.

```python
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - SVM')
plt.show()
```

---
