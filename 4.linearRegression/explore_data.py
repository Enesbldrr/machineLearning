import pandas as pd

df = pd.read_csv('housing.csv')

print('=== İlk 5 Satır ===')
print(df.head(), '\n')

print('=== Veri Bilgisi ===')
print(df.info(), '\n')

print('=== Temel İstatistikler ===')
print(df.describe(), '\n')

print('=== Eksik Değerler ===')
print(df.isnull().sum(), '\n')

# total_bedrooms için medyan ile doldurma
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# ocean_proximity için one-hot encode
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

output_path = 'processed_housing.csv'
df.to_csv(output_path, index=False)
print(f'Ön işleme tamamlandı. Temizlenmiş veri "{output_path}" dosyasına kaydedildi.')