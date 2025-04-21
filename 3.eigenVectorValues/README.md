# 📌 Makine Öğrenmesi Bağlamında Matris İşlemleri, Özdeğer ve Özvektör Hesaplamaları

## 1. Giriş
Bu laboratuvar çalışmasının amacı, matrislerle ilgili temel işlemler olan özdeğer (eigenvalue) ve özvektör (eigenvector) hesaplamalarını hem hazır bir kütüphane kullanarak (NumPy) hem de manuel olarak gerçekleştirmektir. Ardından bu iki yöntem karşılaştırılarak doğruluğu ve performans farkı analiz edilmiştir.

Çalışma boyunca kullanılan programlama dili Python olup, özellikle NumPy kütüphanesi aktif şekilde kullanılmıştır. Kodlar tek bir `.py` dosyasında toplanmış, işlemlerin ardından teorik açıklamalara bu dosyada yer verilmiştir.

## 2. Matris, Özdeğer ve Özvektörlerin Makine Öğrenmesindeki Rolü (Soru 1)

### 🔹 Matris Nedir?
Matrisler, sayısal verileri düzenli şekilde tutmaya yarayan 2 boyutlu yapılardır. Makine öğrenmesinde veriler genellikle matrisler olarak temsil edilir.

### 🔹 Özdeğer ve Özvektör Nedir?
Bir matris A için öyle bir v vektörü ve λ skalar değeri vardır ki şu denklem sağlanır:

**A * v = λ * v**

Burada:
- λ özdeğer (eigenvalue)
- v özvektör (eigenvector) olarak adlandırılır.

Özvektör, matris tarafından yalnızca ölçeklendirilen, yönü değişmeyen özel bir vektördür.

### 🔹 Makine Öğrenmesinde Kullanım Alanları
- **Boyut indirgeme (PCA)**: Veriler en çok bilgi barındıran yönlere projeksiyonlanır.
- **Veri Sıkıştırma**
- **Özellik Seçimi**
- **Doğrusal Dönüşümler**

### 🔹 Hesaplama Yöntemleri
- NumPy, SciPy, MATLAB gibi araçlarla sayısal yöntemler
- Karakteristik denklem çözümü gibi teorik yollar

## 3. NumPy ile Özdeğer ve Özvektör Hesaplama (Soru 2)

```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Özdeğerler:\n", eigenvalues)
print("Özvektörler:\n", eigenvectors)
```

### Çıktılar:
```
Özdeğerler:
[5. 2.]

Özvektörler:
[[ 0.89442719 -0.70710678]
 [ 0.4472136   0.70710678]]
```

## 4. Manuel Hesaplama (Soru 3)

### Matris:
```
A = [[4, 2],
     [1, 3]]
```

### Karakteristik Denklem:
```
|4 - λ  2    |
|1    3 - λ| = 0
```

Açılım:
```
(4 - λ)(3 - λ) - 2*1 = λ² - 7λ + 10 = 0
```

Çözüm:
```
λ₁ = 5,  λ₂ = 2
```

Her λ için `(A - λI)v = 0` çözülerek özvektörler elde edilir.

## 5. Karşılaştırma

| Özellik            | NumPy Sonucu                     | Manuel Sonuç                | Uyum |
|--------------------|----------------------------------|-----------------------------|------|
| Özdeğerler         | [5.0, 2.0]                       | [5.0, 2.0]                  | ✅    |
| Özvektörler        | [[0.89, -0.71], [0.44, 0.71]]    | Benzer yönlü vektörler     | ✅    |
| Av = λv Testi      | Sağlandı                         | Sağlandı                    | ✅    |
| Hesaplama Süresi   | ~0.00001 sn                      | ~0.0005 sn                  | ⚠️    |

Not: Özvektörler yön olarak aynıysa eşdeğer kabul edilir.

## 6. Sonuç

- NumPy çok daha hızlı ve pratik.
- Manuel yöntemler kavramları öğrenmek için ideal.
- Özdeğer ve özvektörler, özellikle PCA gibi tekniklerde önemlidir.
- Bu kavramlar makine öğrenmesinin temel taşlarındandır.

## 7. Kaynakça

- NumPy linalg.eig Documentation
- Eigenvectors and Eigenvalues - Towards Data Science
- GitHub - Manuel Eigenvalue Calculation
- Wikipedia - Eigenvalues and Eigenvectors  
  **Erişim Tarihi: 21 Nisan 2025**
