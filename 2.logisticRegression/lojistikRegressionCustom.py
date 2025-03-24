import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Veriyi yüklüyoruz: 'startup_data_processed.csv' dosyası sadece sayısal sütunlar içermektedir.
df = pd.read_csv('startup_data_processed.csv')

# Özellikler (x) ve hedef değişken (y) belirleniyor.
# 'Profitable' sütunu hedef değişken olarak kullanılıyor, diğer tüm sütunlar özellik.
x = df.drop('Profitable', axis=1)
y = df['Profitable']

# Eğitim ve test setlerine ayırıyoruz (stratify ile sınıf dağılımını koruyoruz)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Verilerin ölçeklendirilmesi: Eğitim verisine uygun scaler oluşturup, hem eğitim hem test verisine uyguluyoruz.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bias terimini modele eklemek için her örneğe 1 sütunu ekliyoruz.
X_train_bias = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_bias = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

# Hedef değişkenleri numpy array formatına çevirip sütun vektörü haline getiriyoruz.
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)

# Başlangıç ağırlıklarını küçük rastgele değerlerle başlatıyoruz.
weights_initial = np.random.randn(X_train_bias.shape[1]) * 0.01
weights_initial = weights_initial.reshape(-1, 1)


# Sigmoid aktivasyon fonksiyonu tanımlanıyor.
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


# Binary Cross Entropy Loss (maliyet) hesaplama fonksiyonu
def compute_cost(x, y, w):
    Z = np.dot(x, w)
    y_pred = sigmoid(Z)
    epsilon = 1e-15  # Log(0) hatalarını önlemek için küçük değer
    cost = y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon)
    return -np.mean(cost)


# Gradient Descent ile ağırlık güncelleme fonksiyonu
def gradient_descent(x, y, w, learning_rate=0.001, n_steps=1000, print_cost=True):
    m = x.shape[0]  # Örnek sayısı
    for i in range(n_steps):
        Z = np.dot(x, w)  # Doğrusal birleşim hesaplanıyor
        y_pred = sigmoid(Z)  # Sigmoid aktivasyonu uygulanıyor
        gradient = np.dot(x.T, y_pred - y) / m  # Ortalama gradyan hesaplanıyor
        w -= learning_rate * gradient  # Ağırlıklar güncelleniyor

        # Her 100 adımda mevcut maliyeti hesaplayıp ekrana yazdırıyoruz
        if print_cost and i % 100 == 0:
            cost = compute_cost(x, y, w)
            print(f"Adım {i}: Maliyet = {cost}")
    return w


# Modelin eğitim (fit) süresini ölçüyoruz
start_fit = time.time()
weights_final = gradient_descent(X_train_bias, y_train, weights_initial,
                                 learning_rate=0.001, n_steps=1000, print_cost=True)
fit_time = time.time() - start_fit

# Test verisi üzerinde tahmin (predict) süresini ölçüyoruz
start_predict = time.time()
y_pred_probs = sigmoid(np.dot(X_test_bias, weights_final))
predict_time = time.time() - start_predict

# Olasılık değerlerine eşik uygulayarak (0.5) sınıflandırma yapıyoruz.
y_preds = (y_pred_probs > 0.5).astype(int)

# Modelin metriklerini hesaplıyoruz
acc = accuracy_score(y_test, y_preds)
prec = precision_score(y_test, y_preds, zero_division=0)
rec = recall_score(y_test, y_preds, zero_division=0)
f1 = f1_score(y_test, y_preds, zero_division=0)

print(f"Accuracy (Doğruluk): {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1 Score: {f1}")
print(f"Model Eğitim Süresi (fit): {fit_time:.4f} saniye")
print(f"Tahmin Yapma Süresi (predict): {predict_time:.4f} saniye")

# Confusion Matrix hesaplanıyor
cm = confusion_matrix(y_test, y_preds)

# Confusion Matrix görselleştirmesi yapılıyor
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Karlı Değil', 'Karlı'])
plt.yticks(tick_marks, ['Karlı Değil', 'Karlı'])

# Hücre değerlerini matrise ekliyoruz
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
