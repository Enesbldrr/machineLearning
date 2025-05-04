import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. İşlenmiş veri setini yükleme
df = pd.read_csv('processed_housing.csv')

# 2. Özellikler ve hedef
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

# 3. Eğitim/test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Test setinde tahmin
y_pred = model.predict(X_test)

# 6. Değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Sklearn LinearRegression Sonuçları:")
print(f"  - MSE: {mse:.4f}")
print(f"  - R2: {r2:.4f}\n")

print("Katsayılar ve Sabit Terim:")
print(f"  - Intercept: {model.intercept_:.4f}")
for name, coef in zip(X.columns, model.coef_):
    print(f"  - {name}: {coef:.4f}")
