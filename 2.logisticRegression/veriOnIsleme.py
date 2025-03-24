import pandas as pd

df = pd.read_csv('startup_data.csv')

# Sadece sayısal sütunları seçiyoruz (int64 ve float64 tipindeki sütunlar)
df_numeric = df.select_dtypes(include=['int64', 'float64'])

output_file = 'startup_data_processed.csv'
df_numeric.to_csv(output_file, index=False)

print(f"İşlenmiş veri başarıyla kaydedildi: {output_file}")
