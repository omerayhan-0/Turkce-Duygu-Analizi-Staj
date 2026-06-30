import re
import nltk
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer
import streamlit as st
import joblib

#Sayfa ayarları
st.set_page_config(page_title="Duygu Analizi V1",layout="centered")

#arayüz tasarımı(büyük başlıklar)
st.title("Türkçe E-Ticaret Duygu Analizi")
st.markdown("Yapay Zeka modelimizle, müşteri yorumlarının Pozitif, Negatif veya Nötr olduğunu anında tespit edin")

st.divider() #araya şık bir çizgi çeker

@st.cache_resource #hafızada saklar, aynı fonksiyon tekrar çağrılırsa hafızadaki sonucu kullan
def load_models():
    yuklenen_sozluk = joblib.load("models/vectorizer.pkl")

    yuklenen_model = joblib.load("models/nb_model.pkl")

    return yuklenen_sozluk, yuklenen_model

vectorizer, nb_model = load_models()

#yorum temizleme motoru
nltk.download('stopwords', quiet=True) #indirme işlemindeki gereksiz yazıları guiet true ile görmeden indirebiliriz.
tr_stopwords = set(stopwords.words('turkish'))
tr_stopwords.update(['bir', 'kadar', 'daha', 'çok', 'en'])
stemmer = TurkishStemmer()

#manuel ekstra sozluk (bazı kelimelerin yazımı yanlış ve henüz zemberek eklenmedi)
yazim_sozlugu = {"her kes": "herkes", "hiç bir": "hiçbir", "her şey": "herşey", "beyendim": "beğendim", "malesef": "maalesef", "yanlız": "yalnız", "herkez": "herkes"}

def metin_on_isleme(text):
    text=str(text).replace('İ','i').replace('I','ı').lower()
    text = re.sub(r'[^\w\s]', ' ',text)
    text = re.sub(r'\d+', '', text)

    for hatali, dogru in yazim_sozlugu.items():
        text = text.replace(hatali,dogru)

    kelimeler = text.split()
    temiz_ve_govde = [stemmer.stem(k) for k in kelimeler if k not in tr_stopwords]
    return " ".join(temiz_ve_govde)


#kullanıcıdan alınacak metin kutusu
user_input = st.text_area("Analiz edilecek yorumu giriniz: ", height=150, placeholder="Örn: Kargo çok hızlı geldi ama ürünün kutusu ezilmişti.")

#Analiz butonu, butona tıklayınca altındakiler çalışır
if st.button("Yorumu Analiz Et", use_container_width=True):
    #kutu boşsa kullanıcıyı uyar
    if user_input.strip() == "":
        st.warning("Lütfen analiz etmek için bir yorum yazın!")
    
    else:
        temiz_yorum = metin_on_isleme(user_input)

        metin_vektoru = vectorizer.transform([temiz_yorum])
        
        
        #modelden tahmin iste(positive,negative,notr)
        tahmin = nb_model.predict(metin_vektoru)[0] 
        #yapay zeka modelleri cevabı bir kutu içinde gönderir. 0 yazmazsak daha sonra if ile kontrol yapamayız. kutuyu açıp bize ilk elemanı verir

        #çıkan sonuca göre ekrana renkli bildirimler bas
        st.subheader("Analiz Sonucu: ")

        if tahmin == "Positive":
            st.success("🟢 Pozitif yorum")

            st.balloons()

        elif tahmin == "Negative":
            st.error("🔴 Negatif yorum")

        else:
            st.info("⚪ Nötr yorum")

st.markdown("---")
st.caption("Geliştirici: Ömer Faruk Ayhan")