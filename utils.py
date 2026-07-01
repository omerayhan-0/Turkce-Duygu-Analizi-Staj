import re
import nltk
from nltk.corpus import stopwords

# Gerekli NLTK dosyalarını arka planda indir
nltk.download('stopwords', quiet=True)
tr_stopwords = set(stopwords.words('turkish'))
tr_stopwords.update(['bir', 'kadar', 'daha', 'çok', 'en'])

# OOV / Manuel Düzeltme Sözlüğü
yazim_sozlugu = {
    "her kes": "herkes", 
    "hiç bir": "hiçbir", 
    "her şey": "herşey", 
    "beyendim": "beğendim", 
    "malesef": "maalesef", 
    "yanlız": "yalnız", 
    "herkez": "herkes"
}

def metin_on_isleme(text):
    # 1. Küçük harf ve boşluk
    text = str(text).replace('İ', 'i').replace('I', 'ı').lower()
    
    # 2. Noktalama ve Rakam temizliği
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # 3. Manuel yazım düzeltmeleri
    for hatali, dogru in yazim_sozlugu.items():
        text = text.replace(hatali, dogru)
        
    # 4. Stopwords temizliği (KÖK BULMA İPTAL EDİLDİ - EKLER KORUNUYOR)
    kelimeler = text.split()
    temiz = [k for k in kelimeler if k not in tr_stopwords]
            
    return " ".join(temiz)
