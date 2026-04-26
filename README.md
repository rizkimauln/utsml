# Citrus Fruit Classification (Orange vs Grapefruit)

Tugas ini disusun untuk memenuhi tugas/Ujian Tengah Semester (UTS) mata kuliah Machine Learning.

## Deskripsi Proyek

Tujuan dari program ini adalah untuk membangun model Machine Learning yang dapat memprediksi apakah sebuah buah sitrus adalah **Orange** (Jeruk) atau **Grapefruit** (Jeruk Bali) berdasarkan fitur-fitur fisik yang terdapat pada dataset seperti diameter dan berat buah.

Tiga algoritma yang diimplementasikan dan dibandingkan performanya dalam proyek ini adalah:
1. **Decision Tree**
2. **Naive Bayes**
3. **Support Vector Machine (SVM)**

## Struktur File dan Direktori

Proyek ini dibagi menjadi beberapa folder berdasarkan model algoritma yang digunakan. Setiap folder berisi source code, visualisasi hasil evaluasi (Confusion Matrix & ROC Curve), serta penjelasan spesifik mengenai model tersebut.

```text
UTS ML/
│
├── Decision_Tree/
│   ├── decision_tree.py
│   ├── README.md
│
├── Naive_Bayes/
│   ├── naive_bayes.py
│   ├── README.md
│
├── Support_Vector_Machine/
│   ├── svm.py
│   ├── README.md
│
├── citrus.csv
├── .gitignore
└── README.md
```

## Dataset

Dataset yang digunakan adalah `citrus.csv`. Dataset ini berisi informasi mengenai buah sitrus dengan beberapa kolom fitur utama seperti:
- `name`: Label target klasifikasi (`orange` atau `grapefruit`)
- `diameter`: Diameter buah (dalam cm)
- `weight`: Berat buah (dalam gram)
- `red`, `green`, `blue`: Nilai intensitas warna RGB dari buah tersebut.

## Cara Menjalankan Proyek

1. **Clone repository ini** ke komputer lokal Anda:
   ```bash
   git clone https://github.com/rizkimauln/utsml.git
   cd utsml
   ```

2. **Pastikan library Python yang dibutuhkan sudah terinstal**. Anda bisa menginstalnya menggunakan pip:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```

3. **Jalankan script model** yang ingin Anda uji. Misalnya, untuk menjalankan model SVM:
   ```bash
   cd Support_Vector_Machine
   python svm.py
   ```
   Atau jalankan dari root direktori:
   ```bash
   python Support_Vector_Machine/svm.py
   ```

## Evaluasi Model

Setiap model dievaluasi menggunakan metrik akurasi, *classification report* (Precision, Recall, F1-Score), serta visualisasi berupa *Confusion Matrix* dan *ROC Curve* untuk melihat seberapa baik model dapat membedakan antara kedua kelas (Orange dan Grapefruit). Penjelasan detail mengenai hasil setiap model dapat dilihat pada file `README.md` di masing-masing folder model.

### Perbandingan Hasil Ketiga Model

Berdasarkan pengujian pada *test set*, berikut adalah ringkasan hasil evaluasi performa dari masing-masing algoritma:

| No | Model | Accuracy | F1-Score |
|:---:|---|:---:|:---:|
| 1 | **Decision Tree** | 93.80% | 0.9380 |
| 2 | **Naive Bayes** | 92.08% | 0.9208 |
| 3 | **Support Vector Machine (SVM)**| 93.68% | 0.9368 |

Dari tabel di atas, dapat dilihat bahwa algoritma **Decision Tree** menghasilkan performa paling tinggi dengan akurasi **93.80%**, sangat bersaing ketat dengan model **SVM** yang memperoleh akurasi **93.68%**. Sementara itu, **Naive Bayes** berada sedikit di bawah keduanya dengan akurasi **92.08%**.
