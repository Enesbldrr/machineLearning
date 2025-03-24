import numpy as np
import pandas as pd

# 1. Orijinal veriyi yükle
data = pd.read_csv('telco.csv')

# 2. Gereksiz kolonları kaldır
data.drop(['Unnamed: 0', 'customerID'], axis=1, inplace=True)

# 3. Categorical / Binary Dönüşümler
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
data['SeniorCitizen'] = data['SeniorCitizen'].map({'Yes': 1, 'No': 0})
data['Partner'] = data['Partner'].map({'Yes': 1, 'No': 0})
data['Dependents'] = data['Dependents'].map({'Yes': 1, 'No': 0})
data['PhoneService'] = data['PhoneService'].map({'Yes': 1, 'No': 0})

# 4. One-Hot Encoding için çok kategorili değişkenler
cols_to_dummy = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
data = pd.get_dummies(data, columns=cols_to_dummy, drop_first=True)

data = pd.get_dummies(data, columns=['Contract'], drop_first=True)
data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes': 1, 'No': 0})
data = pd.get_dummies(data, columns=['PaymentMethod'], drop_first=True)

# 5. Sayısal dönüşümler
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
total_charges_mean = data['TotalCharges'].mean()
data['TotalCharges'].fillna(total_charges_mean, inplace=True)

# 6. Hedef değişken dönüşümü: 'Churn' sütunu
data['Churn'] = data['Churn'].map({'Stayed': 1, 'Churned': 0})

# İşlenmiş veriyi kontrol edelim
print(data.info())
print(data.head())

# 7. İşlenmiş veriyi CSV dosyası olarak kaydet
data.to_csv('telco_processed.csv', index=False)
print("İşlenmiş veri 'telco_processed.csv' olarak kaydedildi.")
