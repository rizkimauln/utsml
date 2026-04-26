# 🤖 Naive Bayes Classification — Oranges vs Grapefruit

Panduan praktikum membangun model klasifikasi menggunakan algoritma **Gaussian Naive Bayes** dengan dataset Oranges vs Grapefruit.

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

> Muat semua library yang dibutuhkan untuk manipulasi data, visualisasi, preprocessing, dan pemodelan machine learning. Tambahan metrics seperti `roc_curve` dan `auc` digunakan untuk visualisasi evaluasi.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
```

---

### 2. Load Data

> Baca dataset dari file CSV yang berada di direktori atas dan tampilkan jumlah datanya.

```python
df = pd.read_csv('../citrus.csv')
print("Data berhasil dimuat. Jumlah baris:", len(df))
```

---

### 3. Preprocessing

> Ubah kolom target (`name`) menjadi numerik untuk diproses. Model matematis membutuhkan representasi angka.

```python
le = LabelEncoder()
df['name'] = le.fit_transform(df['name'])
```

---

### 4. Split Data & Scaling

> Bagi dataset ke dalam Train dan Test menggunakan ukuran 75:25. Skalakan data menggunakan StandardScaler agar memiliki distribusi yang seragam.

```python
X = df.drop(columns=['name']).values
y = df['name'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

---

### 5. Training Model

> Inisialisasi dan latih model **Gaussian Naive Bayes**. Ini mengasumsikan antar fitur bersifat independen dan didistribusi secara normal mengikuti Gaussian distribution.

```python
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

---

### 6. Prediksi

> Terapkan model yang dilatih pada testing data untuk menghasilkan prediksi.

```python
y_pred = classifier.predict(X_test)
```

---

### 7. Evaluasi Model

> Evaluasi performa model menggunakan berbagai metrik seperti Accuracy, F1-Score (macro average), dan melihat Classification Report.

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
```

#### 📌 Visualisasi Confusion Matrix
> Menampilkan heatmap dari tebakan model, lalu menyimpan plot ke dalam file gambar.

```python
cf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_naive_bayes.png')
```

#### 📌 ROC Curve
> Memplot *Receiver Operating Characteristic* untuk Naive Bayes.

```python
y_pred_proba = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Naive Bayes')
plt.legend(loc="lower right")
plt.savefig('roc_curve_naive_bayes.png')
```
