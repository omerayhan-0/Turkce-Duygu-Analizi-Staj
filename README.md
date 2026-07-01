# 🛍️ Türkçe E-Ticaret Duygu Analizi (Sentiment Analysis)

Bu proje, Türkçe e-ticaret yorumlarının duygu durumunu (Pozitif, Negatif, Nötr) sınıflandırmak amacıyla geliştirilmiş, Uçtan Uca (End-to-End) bir Makine Öğrenmesi (NLP) uygulamasıdır. Proje, 45.000 satırlık dengeli bir veri seti kullanılarak eğitilmiş olup kullanıcı dostu bir Streamlit web arayüzü ile sunulmaktadır.

## 🚀 Özellikler

- **Dengeli Veri Seti:** Nötr sınıf zorluklarını aşmak adına her sınıftan (Pozitif, Negatif, Nötr) 15.000'er adet veri çekilerek oluşturulmuş 45.000 yorumluk dengeli veri seti.
- **Akıllı Metin Ön İşleme:** Türkçe'nin sondan eklemeli yapısına uygun olarak kök bulma (stemming) iptal edilmiş ve anlamsal (semantic) bütünlük korunmuştur.
- **Gelişmiş Güvenlik:** 
  - **OOV (Out-of-Vocabulary) Koruması:** Anlamsız (Örn: "123123") veya sözlükte olmayan kelimeler girildiğinde modelin çökmesini veya uydurmasını engelleyen Vektör Toplamı (Zero-Sum) kalkanı.
  - **Güven Skoru (Confidence Score):** Modelin tahmin güvencesini (`predict_proba`) ölçerek, kararında %50'nin altında emin olduğu (Örn: *Kulaklığın tadı güzel*) durumlarda kullanıcıya uyarı (Explainable AI) veren şeffaf karar mekanizması.
- **Modüler Mimari (DRY):** Kod tekrarını önlemek için veri temizleme fonksiyonları `utils.py` isimli merkezi bir dosyada toplanmıştır.

## 🛠️ Kullanılan Teknolojiler

- **Python 3.13**
- **Scikit-Learn:** TF-IDF Vektörizasyonu ve Naive Bayes (MultinomialNB) Sınıflandırma
- **NLTK / RegEx:** Durdurma kelimeleri (Stopwords), noktalama ve rakam temizliği
- **Pandas:** Veri işleme, manipülasyon ve analiz
- **Streamlit:** Web Arayüzü (Frontend)
- **HuggingFace Datasets:** E-ticaret veri setinin çekilmesi

## ⚙️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Depoyu bilgisayarınıza klonlayın:
   ```bash
   git clone https://github.com/omerayhan-0/Turkce-Duygu-Analizi-Staj.git
   cd Turkce-Duygu-Analizi-Staj/Proje
   ```

2. Gerekli kütüphaneleri indirin:
   ```bash
   pip install -r requirements.txt
   ```

3. Modeli ve web arayüzünü başlatın:
   ```bash
   streamlit run app.py
   ```

## 🧠 Model Detayları

- **Algoritma:** Naive Bayes (MultinomialNB)
- **Vektörizasyon:** TF-IDF (1,2 n-gram aralığı ile)
- **Doğruluk Oranı (Accuracy):** ~%77
- Neden Naive Bayes? Veri setindeki 172.000 kelimelik yüksek boyutlu özellik (feature) uzayında çok hızlı ve etkili çalıştığı için seçilmiştir.

## 🧑‍💻 Geliştirici
**Ömer Faruk Ayhan**
Bera Ar-Ge Yazılım - Staj Projesi
