import streamlit as st
import joblib
from utils import metin_on_isleme

# Sayfa ayarları
st.set_page_config(page_title="Duygu Analizi V1", layout="centered")

# Arayüz tasarımı
st.title("🛍️ Türkçe E-Ticaret Duygu Analizi")
st.markdown("Yapay Zeka modelimizle, müşteri yorumlarının Pozitif, Negatif veya Nötr olduğunu anında tespit edin")

st.divider()

@st.cache_resource
def load_models():
    yuklenen_sozluk = joblib.load("models/vectorizer.pkl")
    yuklenen_model = joblib.load("models/nb_model.pkl")
    return yuklenen_sozluk, yuklenen_model

vectorizer, nb_model = load_models()

# Kullanıcıdan alınacak metin kutusu
user_input = st.text_area("Analiz edilecek yorumu giriniz: ", height=150, placeholder="Örn: Kargo çok hızlı geldi ama ürünün kutusu ezilmişti.")

# Analiz butonu
if st.button("Yorumu Analiz Et", use_container_width=True):
    # Kutu boşsa uyar
    if user_input.strip() == "":
        st.warning("Lütfen analiz etmek için bir yorum yazın!")
    

    else:
        # Modüler temizlik motorumuzla metni temizle
        temiz_yorum = metin_on_isleme(user_input)
        
        # Metni matematiğe çevir
        metin_vektoru = vectorizer.transform([temiz_yorum])
        
        #güvenlik önlemi
        if metin_vektoru.sum() == 0:
            st.warning("Girdiğiniz kelimeler (örn: özel isimler) yapay zeka sözlüğümüzde bulunamadı. Lütfen e-ticaret ile ilgili geçerli bir yorum yazın!")
            
        else:
            # Modelden tahmin ve olasılık iste
            tahmin = nb_model.predict(metin_vektoru)[0]
            olasiliklar = nb_model.predict_proba(metin_vektoru)[0]
            en_yuksek_olasilik = max(olasiliklar) * 100
            
            # Sonucu ekrana bas
            st.subheader("Analiz Sonucu:")

            if tahmin == "Positive":
                st.success("🟢 Pozitif yorum")
                st.balloons()
            elif tahmin == "Negative":
                st.error("🔴 Negatif yorum")
            else:
                st.info("⚪ Nötr yorum")
                
            # Güven Skoru Uyarısı
            if en_yuksek_olasilik < 50:
                st.warning(f"Model bu tahminden tam emin değil (Güven Skoru: %{en_yuksek_olasilik:.1f}). Yorum hem olumlu hem olumsuz öğeler içeriyor olabilir.")
            else:
                st.caption(f"Modelin Karar Güveni: %{en_yuksek_olasilik:.1f}")

st.markdown("---")
st.caption("Geliştirici: Ömer Faruk Ayhan")