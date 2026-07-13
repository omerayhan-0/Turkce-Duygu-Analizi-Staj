# 🛍️ Türkçe E-Ticaret Duygu Analizi (BERTürk tabanlı NLP Projesi)

Bu proje, Türkçe e-ticaret yorumlarının duygu durumunu (Pozitif, Negatif, Nötr) sınıflandırmak amacıyla geliştirilmiş, Uçtan Uca (End-to-End) bir Derin Öğrenme (Deep Learning) uygulamasıdır. Proje, Hugging Face **BERTürk** modelinin ince ayar (Fine-Tuning) yöntemiyle 120.000 satırlık devasa ve dengeli bir veri seti üzerinde eğitilmesiyle oluşturulmuş olup, kurumsal bir Streamlit web arayüzü ve interaktif panellerle (Dashboard) sunulmaktadır.

## 🚀 Özellikler

- **Derin Öğrenme Altyapısı (BERTürk):** Klasik algoritmaların sınırları aşılarak, çift anlamlı ve kinayeli cümleleri anlayabilen Transformer tabanlı BERT modeli kullanılmıştır.
- **Dengeli Büyük Veri Seti:** Nötr sınıf zorluklarını aşmak ve ezberlemeyi önlemek adına her sınıftan (Pozitif, Negatif, Nötr) 40.000'er adet veri çekilerek oluşturulmuş 120.000 yorumluk dengeli veri seti.
- **Gelişmiş Veri Ön İşleme:** Türkçe'nin yapısına uygun olarak kök bulma (stemming) işlemi iptal edilmiş, RegEx ve NLTK ile durdurma kelimeleri temizlenmiş ve anlamsal bütünlük korunmuştur.
- **OOV (Out-of-Vocabulary) Koruması:** Sözlükte olmayan veya anlamsız girdilere (Örn: "123123") karşı modelin uydurmasını ve hata vermesini engelleyen güvenlik mekanizması.
- **Toplu Veri Analizi (Batch Processing):** Arayüz üzerinden Excel ve CSV dosyaları yüklenerek saniyeler içinde binlerce yorumun analiz edilmesi sağlanmıştır.
- **Kök Neden Analizi (Word Cloud):** Analiz sonuçlarındaki negatif ve pozitif yorumların odak noktalarını dinamik kelime bulutlarıyla (Word Cloud) görselleştirerek satıcılara İş Zekası (BI) içgörüleri sunar.
- **Şeffaf Yapay Zeka (Explainable AI):** Modelin tahmin güvencesini ölçen "Güven Skoru" (`predict_proba`) eklenerek, sistemin verdiği kararlar açıklanabilir hale getirilmiştir.
- **Aktif Öğrenme (Geri Bildirim):** Kullanıcıların modelin tahminlerine "Doğru/Yanlış" bildirimi yapabilmesi için `session_state` tabanlı özel formlar oluşturulmuştur.
- **UI/UX ve Dashboard:** Analiz sonuçları Plotly grafikleri ile kurumsal bir Sidebar ve Dashboard üzerinden interaktif olarak görselleştirilmiştir. Export özelliğiyle sonuçlar indirilebilir.

## 🛠️ Kullanılan Teknolojiler

- **Python 3.13**
- **Transformers (Hugging Face) & PyTorch:** BERT modelinin indirilmesi, GPU'da eğitilmesi (`Trainer`) ve `pipeline` ile entegrasyonu.
- **Pandas:** Toplu veri işleme ve manipülasyon.
- **Streamlit:** Dinamik web arayüzünün (Frontend) oluşturulması.
- **Plotly:** Etkileşimli pasta ve çubuk grafiklerin (Dashboard) çizdirilmesi.

## ⚙️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda (localhost) çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

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

- **Algoritma:** BERTürk (bert-base-turkish-cased) - Fine Tuned
- **Eğitim Donanımı:** Tesla T4 GPU (Google Colab üzerinden 3 Epoch otonom eğitim)
- **Doğruluk Oranı (Accuracy):** %81.6
- **F1-Skoru:** %81.5
- **Gelişim Süreci:** Projenin başlangıcında Naive Bayes gibi klasik modellerle %77 başarılarda kalınmış, ardından veri seti 120.000 satıra çıkarılarak ölçekleme testleri (Scaling) yapılmış ve Epoch-2'de en iyi genelleştirmeyi yapan Derin Öğrenme mimarisi ana modele (checkpoint-11996) dönüştürülmüştür.

## 🧑‍💻 Geliştirici
**Ömer Faruk Ayhan**  
Bera Ar-Ge Yazılım - Staj Projesi
