import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Veri seti yüklemesi
data = pd.read_csv('telco.csv')

# Veri seti bilgileri
print("İlk veri seti bilgisi:")
print(data.info())
print("\nİlk 5 satır:")
print(data.head())

# İşlevsiz sütunları kaldırma
data.drop(['Unnamed: 0', 'customerID'], axis=1, inplace=True)


# Feature Dönüşümleri
# --- Categorical / Binary Dönüşümler ---
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
data['SeniorCitizen'] = data['SeniorCitizen'].map({'Yes': 1, 'No': 0})
data['Partner'] = data['Partner'].map({'Yes': 1, 'No': 0})
data['Dependents'] = data['Dependents'].map({'Yes': 1, 'No': 0})
data['PhoneService'] = data['PhoneService'].map({'Yes': 1, 'No': 0})

# --- Çok kategorili değişkenler için One-Hot Encoding ---
cols_to_dummy = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
data = pd.get_dummies(data, columns=cols_to_dummy, drop_first=True)

data = pd.get_dummies(data, columns=['Contract'], drop_first=True)
data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes': 1, 'No': 0})
data = pd.get_dummies(data, columns=['PaymentMethod'], drop_first=True)

# --- Sayısal Dönüşümler ---
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
total_charges_mean = data['TotalCharges'].mean()
data['TotalCharges'].fillna(total_charges_mean, inplace=True)

# Hedef değişken: Churn sütunundaki değerler 'Stayed' ve 'Churned'
data['Churn'] = data['Churn'].map({'Stayed': 1, 'Churned': 0})

# Dönüşümden sonra veriseti kontrolü
print("\nDönüştürülmüş veri seti bilgisi:")
print(data.info())
print("\nDönüştürülmüş veri setinin ilk 5 satırı:")
print(data.head())

# Target ve diğer özelliklerin ayrılması
X = data.drop('Churn', axis=1).values  # .values ile numpy array'e çevirdik
y = data['Churn'].values

# Veriyi test ve eğitim için ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nEğitim seti boyutu:", X_train.shape)
print("Test seti boyutu:", X_test.shape)

#########################################
# scikit-learn GaussianNB Modeli
#########################################
sk_model = GaussianNB()

start_time = time.time()
sk_model.fit(X_train, y_train)
sk_training_time = time.time() - start_time
print("\nScikit-learn model eğitim süresi: {:.4f} saniye".format(sk_training_time))

start_time = time.time()
y_pred_sk = sk_model.predict(X_test)
sk_prediction_time = time.time() - start_time
print("Scikit-learn model tahmin süresi: {:.4f} saniye".format(sk_prediction_time))

print("\nScikit-learn Confusion Matrix:")
cm_sk = confusion_matrix(y_test, y_pred_sk)
print(cm_sk)

print("\nScikit-learn Classification Report:")
print(classification_report(y_test, y_pred_sk))


#########################################
# Bizim GaussianNB Modelimiz
#########################################
class MyGaussianNB:
    def __init__(self):
        self.classes_ = None
        self.mean_ = {}
        self.var_ = {}
        self.priors_ = {}
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            X_c = X[y == c]
            self.mean_[c] = np.mean(X_c, axis=0)
            self.var_[c] = np.var(X_c, axis=0)  # population variance
            self.priors_[c] = X_c.shape[0] / X.shape[0]
        return self
    
    def _calculate_log_likelihood(self, class_value, x):
        # mean ve var değerlerini numpy array ve float tipine çeviriyoruz.
        mean = np.array(self.mean_[class_value], dtype=float)
        var = np.array(self.var_[class_value], dtype=float) + 1e-9  # epsilon ekledik

        # x'i de numpy array'e çeviriyoruz (eğer değilse)
        x = np.array(x, dtype=float)
    
        # Her bir feature için log-gaussian likelihood hesaplaması:
        log_term = np.log(2 * np.pi * var)
        log_likelihood = -0.5 * np.sum(log_term) - 0.5 * np.sum(((x - mean) ** 2) / var)
        return log_likelihood



    
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            x = X[i]
            posteriors = {}
            for c in self.classes_:
                # Log prior
                prior = np.log(self.priors_[c])
                # Toplam log likelihood
                log_likelihood = self._calculate_log_likelihood(c, x)
                posteriors[c] = prior + log_likelihood
            # En yüksek posterior değeri veren sınıfı seç
            best_class = max(posteriors, key=posteriors.get)
            y_pred.append(best_class)
        return np.array(y_pred)

# Modeli eğitme kısmı
my_model = MyGaussianNB()
start_time = time.time()
my_model.fit(X_train, y_train)
my_training_time = time.time() - start_time
print("\nKendi yazdığımız model eğitim süresi: {:.4f} saniye".format(my_training_time))

# Tahmin yapma kısmı
start_time = time.time()
y_pred_my = my_model.predict(X_test)
my_prediction_time = time.time() - start_time
print("Kendi yazdığımız model tahmin süresi: {:.4f} saniye".format(my_prediction_time))

print("\nKendi yazdığımız model Confusion Matrix:")
cm_my = confusion_matrix(y_test, y_pred_my)
print(cm_my)

print("\nKendi yazdığımız model Classification Report:")
print(classification_report(y_test, y_pred_my))


#########################################
# Confusion Matrix Görselleştirme Fonksiyonu
#########################################
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False)
    plt.title(title)
    plt.ylabel('Gerçek Label')
    plt.xlabel('Tahmin Edilen Label')
    plt.show()

plot_confusion_matrix(cm_sk, title="Scikit-learn GaussianNB Confusion Matrix")
plot_confusion_matrix(cm_my, title="Kendi Yazdığımız GaussianNB Confusion Matrix")


# Karşılaştırma için metriklerin hesaplanması
from sklearn.metrics import accuracy_score

sk_accuracy = accuracy_score(y_test, y_pred_sk)
my_accuracy = accuracy_score(y_test, y_pred_my)

print("\n--- Model Karşılaştırması ---")
print("Scikit-learn GaussianNB:")
print(f"\tEğitim süresi: {sk_training_time:.4f} saniye")
print(f"\tTahmin süresi: {sk_prediction_time:.4f} saniye")
print(f"\tDoğruluk (Accuracy): {sk_accuracy:.4f}")

print("\nKendi Yazdığımız GaussianNB:")
print(f"\tEğitim süresi: {my_training_time:.4f} saniye")
print(f"\tTahmin süresi: {my_prediction_time:.4f} saniye")
print(f"\tDoğruluk (Accuracy): {my_accuracy:.4f}")
