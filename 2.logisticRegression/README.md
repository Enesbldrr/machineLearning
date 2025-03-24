# Logistic Regression Model Comparison

Bu projede, aynı veri kümesi üzerinde iki farklı lojistik regresyon modeli uygulanmıştır. Biri scikit-learn kütüphanesi kullanılarak, diğeri ise gradient descent yöntemi ile manuel olarak gerçekleştirilmiştir. Aşağıda, her iki yaklaşımın veri ön işleme, model eğitimi, tahmin, metrik hesaplama ve sonuç görselleştirmelerini nasıl ele aldığı anlatılmaktadır.

---

## 1. Veri Kümesi ve Özellikleri

**Veri Kümesi:**  
- **Dosya:** `startup_data_processed.csv`  
- **Özellikler:**  
  - Orijinal veri kümesinde metinsel sütunlar (ör. Startup Name, Industry, Exit Status, Region) mevcuttu.  
  - Ön işleme sonucu yalnızca sayısal sütunlar kalmıştır:  
    - Funding Rounds  
    - Funding Amount (M USD)  
    - Valuation (M USD)  
    - Revenue (M USD)  
    - Employees  
    - Market Share (%)  
    - Year Founded  
    - **Profitable** (Hedef değişken: 0 = Karlı Değil, 1 = Karlı)

Bu veri kümesi, startup’ların finansal durumları ve operasyonel ölçütlerini içermektedir.

---

## 2. Veri Ön İşleme

Her iki modelde de aşağıdaki veri ön işleme adımları uygulanmıştır:

- **Hedef ve Özellik Ayrımı:**  
  - `Profitable` sütunu hedef değişken olarak ayrılmış; diğer tüm sayısal sütunlar özellik olarak kullanılmıştır.

- **Eğitim ve Test Setlerine Bölme:**  
  - `train_test_split` ile veriler eğitim ve test setlerine ayrılmıştır.
  - Stratify kullanılarak sınıf dağılımı korunmuştur.

- **Manuel Modelde Ek Adımlar:**  
  - **Bias Terimi:** Her örneğe sabit terim (bias) eklenmiştir.
  - **Ölçeklendirme:** `StandardScaler` ile veriler normalize edilerek, gradient descent sırasında oluşabilecek sayısal taşmalar (overflow) engellenmiştir.

---

## 3. Model Uygulamaları

### 3.1. Scikit-Learn ile Lojistik Regresyon

- **Uygulama Özeti:**
  - `LogisticRegression` sınıfı kullanılarak model oluşturulmuştur.
  - Modelin eğitim (fit) ve tahmin (predict) metotlarının çalışma süreleri ölçülmüş, sonuçlar konsola yazdırılmıştır.
  - Performans metrikleri: accuracy, precision, recall, ve F1 score hesaplanmıştır.
  - **Confusion Matrix:**  
    - Matris, modelin tahmin ettiği "Karlı" ve "Karlı Değil" sınıflar arasındaki ilişkiyi görsel olarak sunmaktadır.

- **Ölçülen Metrik Değerleri:**

  | Metrik                 | Değer      |
  |------------------------|------------|
  | Accuracy (Doğruluk)    | 0.5400     |
  | Precision              | 0.3750     |
  | Recall                 | 0.2250     |
  | F1 Score               | 0.2812     |
  | Eğitim Süresi (fit)    | 0.0040 s   |
  | Tahmin Süresi (predict)| 0.0000 s   |

### 3.2. Manuel Lojistik Regresyon (Gradient Descent)

- **Uygulama Özeti:**
  - Sigmoid aktivasyon fonksiyonu, binary cross entropy loss ve gradient descent algoritması elle uygulanmıştır.
  - Ağırlıklar, küçük rastgele değerlerle başlatılıp, iteratif olarak güncellenmiştir.
  - Eğitim ve tahmin süreleri ölçülmüştür.
  - **Veri Ölçeklendirme:**  
    - Manuel modelde, overflow problemlerini engellemek için `StandardScaler` ile özellikler normalize edilmiştir.
  - **Confusion Matrix:**  
    - Modelin doğru ve yanlış sınıflandırma sayılarını görsel olarak sunar.

- **Ölçülen Metrik Değerleri:**

  | Metrik                 | Değer                             |
  |------------------------|-----------------------------------|
  | Accuracy (Doğruluk)    | 0.54                              |
  | Precision              | 0.45714285714285713               |
  | Recall                 | 0.37209302325581395               |
  | F1 Score               | 0.41025641025641024               |
  | Eğitim Süresi (fit)    | 0.0000 s (çok kısa ölçüldü)        |
  | Tahmin Süresi (predict)| 0.0000 s                         |

*Not:* Manuel modelde eğitim ve tahmin sürelerinin 0.0000 s olarak gözükmesi, işlemlerin çok hızlı gerçekleştiğini veya ölçüm hassasiyetinin yetersiz olduğunu gösterebilir. Buna ek olarak, metrik değerlerdeki farklılıklar, model parametreleri, öğrenme oranı ve veri ölçeklendirme gibi adımlardaki uygulama farklarından kaynaklanmaktadır.

---

## 4. Confusion Matrix Görselleştirmesi

Her iki modelde de confusion matrix, modelin sınıflandırma performansını görselleştirmek için kullanılmıştır. Görselleştirme şu unsurları içerir:

- **Satırlar:** Gerçek sınıf etiketleri (Karlı Değil, Karlı)
- **Sütunlar:** Modelin tahmin ettiği sınıf etiketleri (Karlı Değil, Karlı)
- **Hücre Değerleri:** Her bir sınıf için doğru (True Positive, True Negative) ve yanlış (False Positive, False Negative) tahmin sayıları
- Renk skalası, doğru tahminleri ve hataları daha net ayırt etmenizi sağlar.

---

## 5. Sonuç ve Karşılaştırma

- **Scikit-Learn Modeli:**  
  - Genel olarak daha optimize edilmiş ve stabil sonuçlar üretmektedir.
  - Eğitim süresi yaklaşık 0.0040 saniye; tahmin süresi çok kısa.
  - Metrik sonuçlar: Accuracy %54, ancak precision, recall ve F1 score değerleri manuel modele göre daha düşük çıkmıştır.
  - Hazır kütüphane fonksiyonları sayesinde, modelin parametre ayarları ve optimizasyonu daha sofistike şekilde gerçekleştirilmiştir.

- **Manuel Model (Gradient Descent):**  
  - Öğrenme algoritmasının temel prensiplerini anlamak açısından faydalıdır.
  - Eğitim ve tahmin süreleri ölçüm hassasiyeti açısından sıfır olarak gözükse de, model performansı metriklerinde daha yüksek precision (yaklaşık 0.4571), recall (yaklaşık 0.3721) ve F1 score (yaklaşık 0.4103) değerleri elde edilmiştir.
  - Veri ölçeklendirme ve bias ekleme gibi ön işleme adımları, manuel modelin stabil çalışmasına önemli katkı sağlamıştır.

Her iki modelin sonuçları, veri ön işleme ve model parametrelerinin doğru ayarlanmasının, lojistik regresyon uygulamalarında ne kadar kritik olduğunu göstermektedir. Kütüphane tabanlı model, daha yaygın olarak gerçek dünya uygulamalarında tercih edilirken, manuel model ise algoritmanın işleyişini anlamak için değerli bir eğitim aracıdır.

---

*Bu README dosyası, projenin amacını, kullanılan veri kümesini, uygulanan ön işleme adımlarını ve iki farklı lojistik regresyon modelinin karşılaştırmasını kapsamaktadır. Her iki yaklaşımın confusion matrix görselleştirmeleri ve performans metrikleri detaylı olarak sunulmuştur.*
