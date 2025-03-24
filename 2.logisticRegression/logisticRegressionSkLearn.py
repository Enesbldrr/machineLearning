import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Veri kümesini yüklüyoruz
df = pd.read_csv('startup_data_processed.csv')

# Özellikler (X) ve hedef değişkeni (y) belirliyoruz
X = df.drop('Profitable', axis=1)  # 'Profitable' dışındaki tüm sayısal sütunlar
y = df['Profitable']

# Eğitim ve test setlerine ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lojistik regresyon modelimizi oluşturuyoruz
model = LogisticRegression(max_iter=1000)

# Modelin fit metodunun çalışma süresini ölçüyoruz
start_time = time.time()
model.fit(X_train, y_train)
fit_time = time.time() - start_time

# Tahmin yapma süresini ölçüyoruz
start_time = time.time()
y_pred = model.predict(X_test)
predict_time = time.time() - start_time

# Confusion Matrix hesaplıyoruz
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix'i görselleştiriyoruz
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Karlı Değil', 'Karlı'])
plt.yticks(tick_marks, ['Karlı Değil', 'Karlı'])
# Her bir hücre için sayıları ekrana yazdırıyoruz
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin Edilen Değer')
plt.tight_layout()
plt.show()

# Metrikleri hesaplıyoruz
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Sonuçları ekrana yazdırıyoruz
print("Model Eğitim Süresi (fit): {:.4f} saniye".format(fit_time))
print("Tahmin Yapma Süresi (predict): {:.4f} saniye".format(predict_time))
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))
