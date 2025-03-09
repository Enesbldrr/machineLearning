# Telco Churn Tahmin Projesi

Bu projede, bir telekomünikasyon şirketinin müşterilerinin hizmetten ayrılıp ayrılmayacağını (churn) tahmin etmek için Naive Bayes modeli kullanılmıştır. Hem scikit-learn kütüphanesinin GaussianNB modeli hem de kendi yazdığımız Naive Bayes modelini eğitip karşılaştırdık.

## 1. Proje Genel Bakışı
- **Amaç:** Müşterilerin hizmetten ayrılma (churn) olasılığını tahmin etmek.
- **Yöntem:** İkili sınıflandırma problemi olarak değerlendirilen proje, hem hazır GaussianNB hem de kendimiz yazdığımız GaussianNB modeli ile gerçekleştirildi.
- **Veri Seti:** Orijinal veri `telco.csv` dosyasında bulunmaktadır.

## 2. Veri Ön İşleme
Veri üzerinde uygulanan ön işleme adımları şu şekildedir:
- **Gereksiz Sütunların Kaldırılması:**  
  - `Unnamed: 0` ve `customerID` sütunları, sonuçları etkilemediği için kaldırılmıştır.
- **Kategorik Değişkenlerin Dönüştürülmesi:**  
  - `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService` ve `PaperlessBilling` sütunları binary dönüşüm ile 1 ve 0'a çevrilmiştir.
  - `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract` ve `PaymentMethod` sütunları One-Hot Encoding yöntemiyle ayrı sütunlara bölünmüştür.
- **Sayısal Dönüşümler:**  
  - `TotalCharges` sütunu sayısal değere dönüştürülmüş ve oluşan boş değerler ilgili sütunun ortalaması ile doldurulmuştur.
- **Hedef Değişken:**  
  - `Churn` sütunu, orijinal veride "Stayed" ve "Churned" olarak yer alırken, model için 1 (Stayed) ve 0 (Churned) olarak kodlanmıştır.
- **İşlenmiş Verinin Kaydedilmesi:**  
  - Uygulanan tüm dönüşümlerden sonra verinin işlenmiş hali `telco_processed.csv` dosyası olarak kaydedilmiştir.
  - İşlenmiş veriyi kaydetmek için ayrı bir python dosyası kullandım ve `veriönişleme.py` dosyasını projeye ekledim.

## 3. Özelliklerin Açıklamaları (Feature Açıklamaları)
Veri setindeki her bir özelliğin ne anlama geldiğini ve modelde nasıl kullanıldığını aşağıda bulabilirsiniz:

- **gender:**  
  Müşterinin cinsiyeti (Male/Female). Binary dönüşüm ile 0 (Male) ve 1 (Female) olarak kodlanmıştır.

- **SeniorCitizen:**  
  Müşterinin yaşlı olup olmadığı bilgisini içerir. 1 (Yes) ve 0 (No) şeklinde kodlanmıştır.

- **Partner:**  
  Müşterinin partnerinin olup olmadığı bilgisi. Binary dönüşüm uygulanarak 1 (Yes) ve 0 (No) değerlerine dönüştürülmüştür.

- **Dependents:**  
  Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı bilgisi. 1 (Yes) ve 0 (No) olarak kodlanmıştır.

- **tenure:**  
  Müşterinin şirkette kalma süresi (ay cinsinden). Sayısal değer olarak doğrudan kullanılmıştır.

- **PhoneService:**  
  Müşterinin telefon servisi olup olmadığı bilgisi. 1 (Yes) ve 0 (No) olarak dönüştürülmüştür.

- **PaperlessBilling:**  
  Müşterinin kağıtsız fatura kullanıp kullanmadığını gösterir. Binary dönüşüm uygulanmıştır.

- **MonthlyCharges:**  
  Aylık ücretlendirme bilgisini içeren sayısal bir özelliktir.

- **TotalCharges:**  
  Müşterinin toplam ücretlendirme bilgisidir. Önce sayısal değere dönüştürülmüş, ardından oluşan boşluklar ortalama değer ile doldurulmuştur.

- **MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies:**  
  Bu sütunlar müşterinin ilgili hizmetleri kullanıp kullanmadığını veya hizmetin tipini belirtir. One-Hot Encoding yöntemi ile her bir kategori ayrı sütun olarak oluşturulmuştur.

- **Contract:**  
  Müşterinin sözleşme tipini (örneğin: month-to-month, one year, two year) belirtir. One-Hot Encoding uygulanmıştır.

- **PaymentMethod:**  
  Ödeme yöntemini gösterir. One-Hot Encoding yöntemi kullanılarak kategoriler ayrı sütunlara ayrılmıştır.

- **Churn:**  
  Hedef değişkendir; müşterinin hizmetten ayrılıp ayrılmayacağını gösterir. "Stayed" 1, "Churned" 0 olarak kodlanmıştır.

## 4. Model Eğitimi ve Performans Ölçümü
Projede iki farklı model eğitilmiştir:
- **scikit-learn GaussianNB:**  
  Hazır kütüphane kullanılarak eğitilen model.
- **Kendi Yazdığımız GaussianNB:**  
  Eğitim aşamasında her sınıf için ortalama, varyans ve öncelik değerlerini hesaplayan, kendi implementasyonumuz olan model.

Her iki model için:
- **Eğitim Süresi:** `time` modülü kullanılarak ölçülmüştür.
- **Tahmin Süresi:** Modelin tahmin süresi ayrıca ölçülmüştür.
- **Değerlendirme Metrikleri:**  
  Accuracy, Precision, Recall ve F1-Score hesaplanarak model performansı değerlendirilmiştir.

## 5. Karmaşıklık Matrisi (Confusion Matrix) ve Görselleştirme
Her iki modelin sonuçları için confusion matrix hesaplanmış ve seaborn kütüphanesi kullanılarak görselleştirilmiştir. Bu matriste:
- **True Positives (TP)**
- **True Negatives (TN)**
- **False Positives (FP)**
- **False Negatives (FN)**
değerleri yer almaktadır. Matrisi görselleştirerek modelin hangi sınıflarda hata yaptığını, hangi sınıfları doğru tahmin ettiğini açıkça gözlemleyebilirsiniz.

## 6. Değerlendirme Metriklerinin Seçimi ve Problem/Sınıf Dağılımı
Değerlendirme metriklerinin seçimi, problem türü ve sınıf dağılımı açısından oldukça kritiktir:
- **Problem Türü:**  
  Bu proje, ikili sınıflandırma problemi olup, sadece doğruluk (accuracy) tek başına yeterli bir gösterge olmayabilir.
- **Sınıf Dağılımı:**  
  Eğer veri setinde sınıflar arasında dengesizlik varsa (örneğin, churn eden müşteriler ile etmeyenler arasında büyük fark varsa), sadece accuracy yanıltıcı olabilir. Bu durumda;
  - **Precision:** Yanlış pozitiflerin azaltılması açısından önemlidir.
  - **Recall:** Yanlış negatiflerin (gerçek churn edenlerin kaçırılması) önlenmesi açısından kritiktir.
  - **F1-Score:** Precision ve Recall arasındaki dengeyi sağlar.
  
Bu nedenle, model performansını değerlendirirken accuracy'nin yanı sıra precision, recall ve F1-score metrikleri de kullanılmıştır.

## 7. Sonuçlar ve Model Karşılaştırması
Her iki modelin eğitim ve tahmin süreleri, metrik değerleri ve confusion matrix’leri karşılaştırılmıştır:
- **scikit-learn GaussianNB:**  
  Daha hızlı eğitim ve tahmin sürelerine sahip, ancak bazı metriklerde belirli farklılıklar gözlemlenmiştir.
- **Kendi Yazdığımız GaussianNB:**  
  Kendi hesaplamalarımızla elde edilen sonuçlar üzerinden performans karşılaştırması yapılmış ve metodoloji farklılıkları tartışılmıştır.

## 8. Sonuç
Proje sonunda, veri ön işleme adımları, model eğitimi ve detaylı performans değerlendirmesi ile churn tahmini için etkili bir yaklaşım geliştirilmiştir. Model değerlendirme metriklerinin seçiminin problem türü ve sınıf dağılımı açısından önemine vurgu yapılmış, böylece dengesiz veri setlerinde hangi metriklerin daha anlamlı olduğu ortaya konmuştur.
