import streamlit as st
from transformers import pipeline
from utils import metin_on_isleme
import csv
import os
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt


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
st.set_page_config(page_title="Türkçe Duygu Analizi | AI Platform", page_icon="🧠", layout="wide")

#Yan menü (Sidebar)
with st.sidebar:
    st.title("AI Duygu Analizi")
    st.caption("Sürüm 1.0 | BERTürk Tabanlı")
    st.divider()
    
    # Kullanım Kılavuzu
    with st.expander("📖 Kullanım Kılavuzu"):
        st.markdown("""
        **Tekli Yorum Analizi:**
        1. Metin kutusuna analiz edilecek yorumu yazın.
        2. "Yorumu Analiz Et" butonuna basın.
        3. Sonuç ekrana gelecektir.
        
        **Toplu Excel/CSV Analizi:**
        1. İçinde metin sütunu bulunan bir `.csv` veya `.xlsx` dosyası yükleyin.
        2. Açılır listeden analiz edilecek sütunu seçin.
        3. "Tümünü Analiz Et" butonuna basın.
        4. Sonuç tablosu ve grafikler otomatik oluşturulacaktır.
        """)
    
    # Sistem Bilgisi
    with st.expander("⚙️ Sistem Bilgisi"):
        st.markdown("""
        - **Model:** BERTürk (dbmdz/bert-base-turkish-cased)
        - **Eğitim Verisi:** 120.000 Türkçe yorum
        - **Sınıflar:** Pozitif, Negatif, Nötr
        - **Güven Eşiği:** %55
        - **Epoch:** 2 (Overfitting önleme)
        """)
    
    st.divider()
    st.caption("Geliştirici: Ömer Faruk Ayhan")
    st.caption("© 2026 - Proje")

#ANASAYFA
st.title("🛍️ Türkçe E-Ticaret Duygu Analizi")
st.markdown("Yapay Zeka modelimizle, müşteri yorumlarının Pozitif, Negatif veya Nötr olduğunu anında tespit edin")

st.divider()

#modeli yükleme
@st.cache_resource
def load_bert_model():
    #pipeline ile modeli ve sözlüğü projemize bağlıyoruz
    return pipeline("text-classification", model="models/bert_duygu_modeli_epoch2", tokenizer="models/bert_duygu_modeli_epoch2")

duygu_analizoru = load_bert_model()

# LABEL_0, LABEL_1, LABEL_2 yi ingilizce metinlere çeviren sözlük (Her iki sekmede de kullanabilmek için global tanımlıyoruz)
label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

#kullanıcının girdiği tab kısmı için yapılan bölme
tab_tekli, tab_toplu = st.tabs(["Tekli yorum analizi", "Toplu Excel/CSV analizi"])

with tab_tekli:

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
            
            #bert pipeline ile tahmin yapma
            sonuc = duygu_analizoru(temiz_yorum)[0]

            # Global label_mapping sözlüğünü kullanıyoruz (yukarıda tanımlı)

            tahmin=label_mapping[sonuc["label"]]
            en_yuksek_olasilik = sonuc["score"]*100
                
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

#kullanıcının yükleyeceği excel veya csv dosyası için olacak bölme
with tab_toplu:
    st.subheader("Toplu Yorum Analizi (Batch Processing)")
    st.info("İçinde 'yorum' sütunu bulunan bir CSV veya Excel dosyası yükleyin. Sistem tüm yorumları analiz edip size sonuç raporu sunacaktır")

    yuklenen_dosya = st.file_uploader("Dosyanızı yükleyin", type=["csv", "xlsx"])

    if yuklenen_dosya is not None:
        try:
            #excel mi csv mi olduğunu anlayıp okuma
            if yuklenen_dosya.name.endswith('.csv'):
                df_toplu = pd.read_csv(yuklenen_dosya)

            else:
                df_toplu = pd.read_excel(yuklenen_dosya)

            #dosyadaki tüm sütun başlıklarını bul ve kullanıcıya seçtir
            sutunlar = df_toplu.columns.tolist()
            secilen_sutun = st.selectbox("Lütfen analiz edilecek metinlerin bulunduğu sütunu seçiniz:", sutunlar)

            if st.button("Tümünü analiz et", key="toplu_analiz_butonu"):
                with st.spinner("Yapay zeka yorumları okuyor lütfen bekleyin"):
                    sonuclar = []
                    guven_skorlari = []

                    #kullanıcının seçtiği sütunu okuyoruz
                    for yorum in df_toplu[secilen_sutun]:
                        # 1. Kontrol: Eğer veri gerçekten boşsa (Pandas NaN ise) veya boşluksa
                        if pd.isna(yorum) or str(yorum).strip().lower() in ["nan", "none", ""]:
                            sonuclar.append("Anlamsız (Geçersiz Veri)")
                            guven_skorlari.append(0.0)
                        else:
                            temiz_yorum_toplu = metin_on_isleme(str(yorum))

                            # Temizlik sonrası boş kaldıysa
                            if temiz_yorum_toplu.strip() == "":
                                sonuclar.append("Anlamsız (Geçersiz Veri)")
                                guven_skorlari.append(0.0)
                            else:
                                # BERT tahmini
                                tahmin_toplu = duygu_analizoru(temiz_yorum_toplu)[0]
                                guven_skoru = tahmin_toplu['score'] * 100

                                # 2. Kontrol: Aşırı düşük güven skorlarını filtrele
                                if guven_skoru < 55:
                                    sonuclar.append("Anlamsız (Düşük Güven)")
                                    guven_skorlari.append(guven_skoru)
                                else:
                                    sonuclar.append(label_mapping[tahmin_toplu['label']])
                                    guven_skorlari.append(guven_skoru)

                    df_toplu['Yapay_Zeka_Karari'] = sonuclar
                    df_toplu['Guven_Skoru (%)'] = [f"%{x:.1f}" for x in guven_skorlari]
                    
                    st.success("Tüm analizler tamamlandı!")
                    
                    # Hızlı özet kısmı
                    toplam_yorum = len(df_toplu)
                    gecerli_sayisi = len([s for s in sonuclar if 'Anlamsız' not in s])
                    anlamsiz_sayisi = toplam_yorum - gecerli_sayisi
                    pozitif_sayisi = sonuclar.count('Positive')
                    negatif_sayisi = sonuclar.count('Negative')
                    notr_sayisi = sonuclar.count('Neutral')
                    
                    st.divider()
                    met1, met2, met3, met4, met5 = st.columns(5)
                    met1.metric("📝 Toplam Yorum", toplam_yorum)
                    met2.metric("✅ Geçerli", gecerli_sayisi)
                    met3.metric("🟢 Pozitif", pozitif_sayisi)
                    met4.metric("🔴 Negatif", negatif_sayisi)
                    met5.metric("⚪ Nötr / 🚫 Anlamsız", f"{notr_sayisi} / {anlamsiz_sayisi}")
                    
                    # Sonuç tablosunu ekrana bas
                    st.divider()
                    st.dataframe(df_toplu, use_container_width=True)

                    #sonuçları csv olarak indirme
                    csv_verisi = df_toplu.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Sonuçları CSV olarak indir",
                        data=csv_verisi,
                        file_name="duygu_analizi_sonucları.csv",
                        mime="text/csv" #dosya kimlik kartı
                    )
                    
                    # Dashboard GRAFİK kısmı
                    st.divider()
                    st.subheader("📊 Analiz İstatistikleri")
                    
                    # Sadece geçerli tahminleri (Anlamsız yazmayanları) grafik yapalım
                    gecerli_tahminler = df_toplu[~df_toplu['Yapay_Zeka_Karari'].str.contains('Anlamsız')]
                    
                    if not gecerli_tahminler.empty:
                        # Sayfayı iki eşit sütuna böl
                        col_grafik1, col_grafik2 = st.columns(2)
                        
                        # Duyguların sayısını hesapla
                        duygu_dagilimi = gecerli_tahminler['Yapay_Zeka_Karari'].value_counts().reset_index()
                        duygu_dagilimi.columns = ['Duygu', 'Sayı']
                        
                        with col_grafik1:
                            # 1. Grafik: Pasta Grafiği (Genel Dağılım)
                            fig_pie = px.pie(duygu_dagilimi, values='Sayı', names='Duygu', title='Yorumların Genel Duygu Dağılımı', color='Duygu', color_discrete_map={'Positive':'#00cc96', 'Negative':'#ef553b', 'Neutral':'#636efa'})
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                        with col_grafik2:
                            # 2. Grafik: Sütun Grafiği (Sayısal Karşılaştırma)
                            fig_bar = px.bar(duygu_dagilimi, x='Duygu', y='Sayı', title='Sayısal Karşılaştırma', color='Duygu', color_discrete_map={'Positive':'#00cc96', 'Negative':'#ef553b', 'Neutral':'#636efa'})
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        #KELİME BULUTU
                        st.divider()
                        st.subheader("Yorumların Odak Noktaları")

                        col_wc1, col_wc2 = st.columns(2)

                        #Negatif Kelime Bulutu
                        with col_wc1:
                            st.markdown("##### 🔴 Negatif Yorumların Temel Nedenleri")
                            negatif_yorumlar = gecerli_tahminler[gecerli_tahminler['Yapay_Zeka_Karari'] == 'Negative']

                            if not negatif_yorumlar.empty:
                                #tüm negatif yorumları dev metne çeviriyoruz
                                tum_negatif_metin = " ".join(str(yorum) for yorum in negatif_yorumlar[secilen_sutun])
                                wc_negatif = WordCloud(width=400, height=300, background_color='black', colormap='Reds').generate(tum_negatif_metin)

                                fig_neg, ax_neg = plt.subplots(figsize = (6,4))
                                ax_neg.imshow(wc_negatif, interpolation='bilinear')
                                ax_neg.axis('off')
                                st.pyplot(fig_neg)
                            else:
                                st.info("Negatif yorum bulunamadı.")

                        #Pozitif Kelime Bulutu
                        with col_wc2:
                            st.markdown("##### 🟢 Pozitif Yorumların Öne Çıkanları")
                            pozitif_yorumlar = gecerli_tahminler[gecerli_tahminler['Yapay_Zeka_Karari'] == 'Positive']

                            if not pozitif_yorumlar.empty:
                                #Tüm pozitif yorumları tek bir dev metne çeviriyoruz
                                tum_pozitif_metin = " ".join(str(yorum) for yorum in pozitif_yorumlar[secilen_sutun])
                                wc_pozitif = WordCloud(width=400, height=300, background_color='black', colormap='Greens').generate(tum_pozitif_metin)

                                fig_pos, ax_pos = plt.subplots(figsize = (6,4))
                                ax_pos.imshow(wc_pozitif, interpolation='bilinear')
                                ax_pos.axis('off')
                                st.pyplot(fig_pos)

                    else:
                        st.info("Grafik çizmek için yeterli (anlamlı) yorum bulunamadı.")
                        
        except Exception as e:
            st.error(f"Dosya okunurken bir hata oluştu: {e}")