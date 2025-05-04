# 4.LinearRegression

## Amaç

California Housing veri seti kullanılarak iki farklı lineer regresyon yöntemi uygulanmıştır:  
1. **Manuel Normal Denklem (Least Squares)**  
   - Θ = (XᵀX)⁻¹ Xᵀy formülünü doğrudan NumPy ile hayata geçirdik.  
2. **Scikit-learn `LinearRegression`**  
   - Aynı veri üzerinde kütüphanenin closed-form çözümünü kullandık.  

Bu sayede hem temel formülün nasıl çalıştığını kavradık hem de pratik bir kütüphane implementasyonu ile sonuçları karşılaştırdık.

---

## Sonuçlar ve Karşılaştırma

| Yöntem             | MSE              | R²      |
|--------------------|------------------|---------|
| Manuel Denklem     | 4.9084767211e+09 | 0.6254  |
| Scikit-learn       | 4.9084767212e+09 | 0.6254  |

> **Gözlem:** İki yöntemde de elde edilen MSE ve R² değerleri eşleşti. Bu, normal denklemle hesaplanan Θ değerlerinin kütüphanedeki yöntemle aynı olduğunu doğrular.

### Katsayılar (Θ)

| Özellik                        | Katsayı      |
|--------------------------------|--------------|
| Intercept                      | –2 275 547.4 |
| longitude                      | –26 838.3    |
| latitude                       | –25 468.4    |
| housing_median_age             | 1 102.2      |
| total_rooms                    | –6.0         |
| total_bedrooms                 | 102.8        |
| population                     | –38.2        |
| households                     | 48.3         |
| median_income                  | 39 474.0     |
| ocean_proximity_INLAND         | –39 786.7    |
| ocean_proximity_ISLAND         | 136 125.1    |
| ocean_proximity_NEAR BAY       | –5 136.6     |
| ocean_proximity_NEAR OCEAN     | 3 431.1      |

> **Yorum:**  
> - En güçlü pozitif etki `median_income` değişkeninde.  
> - Koordinatlar (`longitude`, `latitude`) negatif katsayılarla coğrafi konumun fiyatı etkilediğini gösteriyor.  
> - “ISLAND” kategorisi ev fiyatını ciddi şekilde artırıyor.

---

## Sonuç ve İleriki Adımlar

Bu çalışma, normal denklem ve kütüphane yönteminin tamamen aynı sonuçları ürettiğini gösterdi.  
İlerleyen aşamalarda şunlar değerlendirilebilir:  
- Özellik ölçeklendirme ve gradient-descent tabanlı optimizasyon  
- Regularizasyon yöntemleri (Ridge, Lasso)  
- Ağaç-tabanlı ve polinomsal modellerle genişletme  

---

## 📋 Genel Değerlendirme ve Özet

1. **Veri Ön İşleme:**  
   - `total_bedrooms`’daki 207 eksik değer medyan ile dolduruldu.  
   - `ocean_proximity` one-hot encode edildi.  

2. **Manuel Normal Denklem**  
   - Θ = (XᵀX)⁻¹Xᵀy formülüyle NumPy’den doğrudan hesaplandı.  
   - Elde edilen MSE ≈ 4.908×10⁹, R² ≈ 0.6254.  

3. **Scikit-learn `LinearRegression`**  
   - Aynı veri ile closed-form çözüm kullanıldı.  
   - Performans (MSE, R²) manuel yöntemle tamamen aynı.  

4. **Sonuçlar**  
   - Özellik katsayıları %100 örtüşüyor: matematiksel formül ve kütüphane implementasyonu eşdeğer.  
   - En güçlü etki `median_income`’da, coğrafi konum ve “ISLAND” kategorisi de mantıklı işaretler taşıyor.  

5. **İleriki Adımlar**  
   - Özellik ölçeklendirme & gradient-descent optimizasyonu  
   - Ridge, Lasso gibi regularizasyon  
   - Ağaç-tabanlı ve polinomsal modellerle genişletme  

> **Not:** Bu proje, temel lineer regresyonun hem teorik hem de pratik tarafını net şekilde kavramamıza olanak sağladı. 😊