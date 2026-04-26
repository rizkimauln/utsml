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

> Ekstrak perkakas library yang dibutuhkan dalam pembentukan klasifikasi menggunakan Support Vector Machines. Tambahan metrics seperti `roc_curve` dan `auc` digunakan untuk visualisasi evaluasi.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
```

---

### 2. Load Data

> Ambil dan baca set data menggunakan framework pandas.

```python
df = pd.read_csv('../citrus.csv')
print("Data berhasil dimuat. Jumlah baris:", len(df))
```

---

### 3. Preprocessing

> Gunakan label encoding dari sklearn untuk memformat kolom kategorikal `name` yang mengandung nilai teks, ke dalam tipe bilangan bulat yang terbaca algoritma.

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

---

### 5. Training Model

> Lakukan inisialisasi pada Classifier SVC. Kita menggunakan fungsi kemiringan Kernel `rbf` dan menambahkan `probability=True` agar kita bisa memplot *ROC Curve* nantinya.

```python
classifier = SVC(kernel='rbf', random_state=42, probability=True)
classifier.fit(X_train, y_train)
```

---

### 6. Prediksi

> Simulasikan prediksi terhadap variabel `X` hasil pengujian.

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
> Memplot heat map matrix membuktikan persebaran predikasi yang salah dan yang tepat pada kedua label buah, dan menyimpannya sebagai file gambar.

```python
cf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_svm.png')
```

#### 📌 ROC Curve
> Memplot *Receiver Operating Characteristic* untuk algoritma SVM untuk mengevaluasi trade-off probabilitas antar kelas.

```python
y_pred_proba = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - SVM')
plt.legend(loc="lower right")
plt.savefig('roc_curve_svm.png')
```
