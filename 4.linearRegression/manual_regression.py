import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. İşlenmiş veri setini yükleme
df = pd.read_csv('processed_housing.csv')

# 2. Özellikler ve hedef (float tipine cast ediyoruz)
X = df.drop(columns=['median_house_value']).values.astype(float)  # (20640, n_features)
y = df['median_house_value'].values.reshape(-1, 1).astype(float)  # (20640, 1)

# 3. Eğitim/test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Bias sütunu ekleme
m_train = X_train.shape[0]
X_train_bias = np.hstack([np.ones((m_train, 1)), X_train])

m_test = X_test.shape[0]
X_test_bias = np.hstack([np.ones((m_test, 1)), X_test])

# 5. Normal denklem ile θ hesaplama: θ = (XᵀX)⁻¹ Xᵀy
XtX = X_train_bias.T.dot(X_train_bias)
XtX_inv = np.linalg.inv(XtX)    # Artık float64 matris
Xty = X_train_bias.T.dot(y_train)
theta = XtX_inv.dot(Xty)

# 6. Test setinde tahmin
y_pred = X_test_bias.dot(theta)

# 7. Değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Manuel Least Squares Sonuçları:")
print(f"  - MSE: {mse:.4f}")
print(f"  - R2: {r2:.4f}\n")

# 8. θ değerlerini (Intercept ve katsayılar) yazdırma
print("Theta (Intercept ve Özellik Katsayıları):")
print(f"  - Intercept: {theta[0][0]:.4f}")
for i in range(1, theta.shape[0]):
    print(f"  - Feature {i}: {theta[i][0]:.4f}")
