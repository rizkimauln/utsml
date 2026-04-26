#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc

print("Memulai proses untuk Model Naive Bayes...")

#%%
# 1. Load Data
df = pd.read_csv('../citrus.csv')
print("Data berhasil dimuat. Jumlah baris:", len(df))

#%%
# 2. Preprocessing
le = LabelEncoder()
df['name'] = le.fit_transform(df['name'])

#%%
# 3. Split Data & Scaling
X = df.drop(columns=['name']).values
y = df['name'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(f"Proporsi data latih: {len(X_train)}, data uji: {len(X_test)}")

#%%
# 4. Training Model
classifier = GaussianNB()
classifier.fit(X_train, y_train)
print("Training model selesai.")

#%%
# 5. Prediksi
y_pred = classifier.predict(X_test)

#%%
# 6. Evaluasi Model
print("\n=== Evaluasi Model Naive Bayes ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

#%%
# Visualisasi Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_naive_bayes.png')
print("Visualisasi confusion matrix disimpan sebagai 'confusion_matrix_naive_bayes.png'")

#%%
# ROC Curve
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
print("Visualisasi kurva ROC disimpan sebagai 'roc_curve_naive_bayes.png'")

# %%
