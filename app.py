import streamlit as st
import joblib
from utils import metin_on_isleme
import csv
import os


def geri_bildirim_kaydet(yorum, modelin_tahmini, gercek_duygu):
    dosya_adi = 'hatali_tahminler.csv'
    dosya_var_mi = os.path.isfile(dosya_adi)

    with open(dosya_adi, mode='a', newline='', encoding='utf-8') as dosya:
        yazici = csv.writer(dosya)

        #eğer dosya ilk defa oluşuyorsa en üste başlık yaz
        if not dosya_var_mi:
            yazici.writerow(['yorum','modelin_tahmini','gercek_duygu'])
        
        #kullanıcının geri bildirimini yeni satır olarak ekle
        yazici.writerow([yorum, modelin_tahmini, gercek_duygu])

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
    st.session_state.analiz_izni = True

if st.session_state.get("analiz_izni", False):
    #kutu boşsa uyar
    if user_input.strip() == "":
        st.warning("Lütfen analiz etmek için bir yorum yazın")
    

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

            #aktif öğrenme butonları
            st.divider()
            st.write("Modelin tahmini doğru mu ?")

            #yan yana 2 tane sütun (kolon) oluşturuyoruz.
            col1, col2 = st.columns(2)

            #1.kolon doğru butonu
            with col1:
                if st.button("Doğru bildin", use_container_width=True):
                    st.success("Güzel, onayınız modelimizin kendine güvenini arttırdı.")

            #2.kollon: yanlış butonu ve düzeltme menüsü
            with col2:
                #kullanıcı yanlış derse aşağı doğru açılan bir menü ortaya çıkacak(expander)
                with st.expander("Yanlış bildin (Modeli Eğit)"):
                    with st.form("geri_bildirim_formu"):
                        gercek_secim = st.selectbox("Sizce doğrusu neydi?", ["Positive", "Negative", "Neutral", "Belirsiz"])

                        #formun kendine özel gönder butonuna basılana kadar sayfayı yenilemez
                        submit = st.form_submit_button("Hatayı gönder")

                        if submit:
                            geri_bildirim_kaydet(user_input, tahmin, gercek_secim)
                            st.success("Geri bildirim kaydedildi")



st.markdown("---")
st.caption("Geliştirici: Ömer Faruk Ayhan")