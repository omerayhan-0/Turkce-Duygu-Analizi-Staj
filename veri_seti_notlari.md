# Veri Seti ve Çekme İşlemi Notları

## Hedef
Hugging Face üzerinden e-ticaret yorumlarından oluşan ve 150.000 yorum barındıran **`maydogan/Turkish_SentimentAnalysis_TRSAv1`** veri setini kullanacağız.
Bu veri seti 3 sınıftan (Positive, Negative, Neutral) oluşmaktadır ve Nötr sınıfı Wikipedia cümleleri değil, gerçek nötr yorumlar içerir.

Veri setindeki her sınıftan tam 15.000'er adet veri çekerek toplam 45.000 yorumdan oluşan dengeli (balanced) azımsanmayacak büyüklükte bir veri seti oluşturacağız.

## Kullanılacak Kütüphaneler
- `datasets` (Hugging Face'ten veriyi indirmek için)
- `pandas` (Veriyi işlemek, gruplamak, karıştırmak ve CSV'ye kaydetmek için)

## Geliştirme Beklentisi
Bu aşamada `1_veri_cekme.ipynb` dosyası içerisinde Python kodu yazılarak yukarıda belirtilen hedef gerçekleştirilecektir. 
*(Kodlar öğrenme amacıyla stajyer tarafından yazılacaktır. Mentör sadece takılınan yerlerde ipucu verecektir.)*
