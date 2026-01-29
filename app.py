import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Sentimen TikTok - Skripsi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI LOAD ASSETS (CACHED) ---
@st.cache_resource
def load_assets():
    try:
        # Download stopwords jika belum ada
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Load Model & Vectorizer
        model = joblib.load('svm_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_data():
    try:
        try:
            df = pd.read_excel('data_bersih.xlsx')
        except:
            df = pd.read_csv('data_bersih.csv')
        return df
    except:
        return None

# --- KONFIGURASI KAMUS DATA ---

# 1. Normalisasi Slang (Updated)
norm_dict = {
    'ust': 'ustadz', 'ustd': 'ustadz', 'tad': 'ustadz', 'ml': 'mobile legends', 'ustad': 'ustadz',
    'mm': 'marksman', 'roam': 'roamer', 'tenk': 'tank', 'ks': 'nyampah',
    'by one': 'duel', 'mechanic': 'mekanik', 'op': 'kuat', 'feed': 'beban',
    'yg': 'yang', 'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'gk': 'tidak',
    'bgt': 'banget', 'sdh': 'sudah', 'kalo': 'kalau', 'tp': 'tapi',
    'jd': 'jadi', 'krn': 'karena', 'd': 'di', 'dr': 'dari', 'dgn': 'dengan',
    'utk': 'untuk', 'org': 'orang', 'sy': 'saya', 'gw': 'gue', 'lu': 'kamu',
    'tau': 'tahu', 'ni': 'ini', 'tu': 'itu', 'sm': 'sama', 'aja': 'saja',
    'emg': 'memang', 'anjir': 'anjing', 'njir': 'anjing', 'cok': 'cok',
    'wkwk': 'tertawa', 'bntu': 'bantu', 'minion': 'prajurit kecil',
    'turret': 'menara', 'lord': 'boss besar', 'asw': 'anjing',
    "bgt": "banget", "bngt": "banget", "bgd": "banget", "bngd": "banget",
    "btw": "ngomong-ngomong", "wkwk": "haha", "wkwwk": "haha", "wkwkwk": "haha",
    "wkakaka": "haha", "hehe": "haha", "huhu": "sedih", "huft": "lelah",
    "anj": "anjing", "anjay": "anjing", "ngntd": "ngentot", "ngtnd": "ngentot",
    "ngentd": "ngentot", "mmk": "memek", "ktl": "kontol", "kontl": "kontol",
    "bangsd": "bangsat", "bngsd": "bangsat", "bngsat": "bangsat", "asu": "anjing",
    "gblk": "goblok", "gblg": "goblok", "gblokk": "goblok", "toll": "tolol",
    "tolll": "tolol", "tololl": "tolol", "tololll": "tolol", "kampng": "kampung",
    "kampret": "kampret", "tai": "tai", "tahi": "tai", "pntd": "pantat",
    "pantek": "pantat", "push": "dorong", "rank": "peringkat", "mabar": "main bareng",
    "ulti": "ultimate", "skill": "kemampuan", "feed": "mati terus", "feeder": "pemain mati terus",
    "noob": "pemula", "pro": "pemain hebat", "afk": "keluar dari permainan", "lag": "lambat",
    "hero": "pahlawan", "match": "pertandingan", "epic": "peringkat epic",
    "mythic": "peringkat mythic", "gm": "peringkat grandmaster", "buff": "peningkatan kekuatan",
    "nerf": "penurunan kekuatan", "damage": "kerusakan", "tank": "penahan serangan",
    "mage": "penyihir", "assassin": "pembunuh", "fighter": "petarung",
    "support": "pendukung", "jungler": "pemburu hutan", "exp": "pengalaman",
    "base": "markas", "mid": "jalur tengah", "bot": "jalur bawah", "top": "jalur atas",
    "draft": "pemilihan hero", "ban": "larangan hero", "pick": "pilihan hero",
    "gg": "bagus sekali", "ez": "mudah", "wp": "kerja bagus", "op": "terlalu kuat",
    "ngga": "tidak", "kok": "mengapa", "dong": "", "nih": "ini", "yaampun": "astaga",
    "yaallah": "ya allah", "astaga": "astaga", "pls": "tolong", "plis": "tolong",
    "sumpah": "sungguh", "parah": "buruk sekali", "kocak": "lucu", "lucu bgt": "lucu banget",
    "auto": "langsung", "real": "nyata", "literally": "benar-benar", "fr": "benar-benar",
    "capek": "lelah", "malu2in": "memalukan", "idk": "i dont know", "imo": "in my opinion",
    "imho": "in my humble opinion", "ikr": "i know right", "tbh": "to be honest",
    "omg": "oh my god", "wtf": "kata kasar", "wth": "apa-apaan", "lmao": "tertawa",
    "lol": "tertawa", "rofl": "tertawa", "smh": "geleng kepala", "brb": "segera kembali",
    "afaik": "sejauh yang saya tahu", "thx": "terima kasih", "ty": "terima kasih",
    "plz": "tolong", "u": "kamu", "ur": "punyamu", "r": "adalah", "ya": "iya",
    "yea": "iya", "yeah": "iya", "fuq": "kata kasar", "fck": "kata kasar",
    "fkn": "kata kasar", "fkng": "kata kasar", "fckn": "kata kasar", "stfu": "diam",
    "bs": "omong kosong", "asf": "sekali", "mf": "kata kasar", "ffs": "kata kasar",
    "gua": "saya", "gue": "saya", "loe": "kamu", "elo": "kamu", "bro": "saudara",
    "sis": "saudari", "sob": "teman", "cuan": "untung", "ngegas": "marah",
    "baper": "terbawa perasaan", "mager": "malas gerak", "gabut": "tidak ada kerjaan",
    "nganggur": "tidak bekerja", "santuy": "santai", "sabi": "bisa", "wkwkw": "haha",
    "anjss": "kata kasar", "anjg": "kata kasar",
}

# 2. Sampah Visual (Daftar Kata Pengganggu untuk WordCloud)
sampah_visual = {
    'nya', 'sih', 'kok', 'mah', 'deh', 'dong', 'kah', 'ni', 'tu', 'tuh',
    'yah', 'wow', 'dan', 'yang', 'di', 'itu', 'ini', 'ke', 'dari',
    'yg', 'ga', 'gak', 'nggak', 'udah', 'sdh', 'dah', 'sy', 'gw',
    'banget', 'bgt', 'dr', 'kalo', 'klu', 'utk', 'ya', 'yu', 'y','iya','gimana',
    'aja', 'saja', 'juga', 'jg', 'km', 'ak', 'aku', 'kami', 'kita',
    'game', 'games', 'gim', 'main', 'maen', 'play', 'player',
    'moonton', 'montoon', 'mobile', 'legends' ,'pa','sia','ko','klau','gua','anj','orang','tpi','tapi','kon','g','ku','jdi','la','kn','kan',
    'biar','klo','gitu','gua','pake','karna','anj','anjing','dpt','gue','gini','tpi','bgt', 'bngt', 'bgd', 'bngd', 'btw', 'wkwk', 'wkwwk', 'wkwkwk', 'wkakaka',
    'hehe', 'huhu', 'huft', 'anj', 'anjir', 'anjay', 'njir', 'ngntd', 'ngtnd',
    'ngentd', 'mmk', 'ktl', 'kontl', 'bangsd', 'bngsd', 'bngsat', 'asu',
    'gblk', 'gblg', 'gblokk', 'toll', 'tolll', 'tololl', 'tololll', 'kampng',
    'kampret', 'tai', 'tahi', 'pntd', 'pantek', 'push', 'rank', 'mabar',
    'ulti', 'skill', 'feed', 'feeder', 'noob', 'pro', 'afk', 'lag', 'ml',
    'hero', 'match', 'epic', 'mythic', 'gm', 'lord', 'minion', 'buff', 'nerf',
    'damage', 'tank', 'mm', 'mage', 'assassin', 'fighter', 'support', 'jungler','lo','roamer',
    'exp', 'turret', 'base', 'mid', 'bot', 'top', 'draft', 'ban', 'pick',
    'gg', 'ez', 'wp', 'op', 'ga', 'gak', 'gk', 'nggak', 'ngga', 'aja', 'kok',
    'dong', 'nih', 'yaampun', 'yaallah', 'astaga', 'pls', 'plis', 'sumpah',
    'parah', 'kocak', 'lucu', 'auto', 'real', 'literally', 'fr', 'capek',
    'malu2in', 'idk', 'imo', 'imho', 'ikr', 'tbh', 'btw', 'omg', 'wtf', 'wth',
    'lmao', 'lol', 'rofl', 'smh', 'brb', 'afaik', 'thx', 'ty', 'pls', 'plz',
    'u', 'ur', 'r', 'ya', 'yea', 'yeah', 'fuq', 'fck', 'fkn', 'fkng', 'fckn',
    'stfu', 'bs', 'asf', 'mf', 'ffs', 'gw', 'gua', 'gue', 'loe', 'lu', 'elo',
    'bro', 'sis', 'sob', 'cuan', 'ngegas', 'baper', 'mager', 'gabut', 'nganggur',
    'santuy', 'sabi', 'wkwkw', 'anjss', 'tololl', 'anjg', 'pa', 'sia', 'ko', 'klau',
    'biar', 'gitu', 'pake', 'karna', 'dpt', 'gini', 'kon', 'ku', 'jdi', 'la', 'kn', 'kan',
    'videonya', 'bang', 'kak', 'mas', 'pak', 'om', 'kan', 'lah', 'pun',
    'bange', 'hukumnya', 'haha', 'kali', 'kau', 'otw', 'trs', 'terus', 'trus',
    'gmna', 'gimana', 'sampe', 'sampai', 'kaya', 'kayak', 'nge', 'emang', 'mang'
}

# --- FUNGSI STOPWORDS ---
def get_custom_stopwords():
    # Load stopwords indo bawaan
    try:
        stop_factory = set(stopwords.words('indonesian'))
    except:
        nltk.download('stopwords')
        stop_factory = set(stopwords.words('indonesian'))

    # A. Stopwords Umum & Sampah Visual
    # Gabungkan stopwords bawaan + sampah visual yang Anda berikan
    general_stop = set(STOPWORDS)
    general_stop.update(stop_factory)
    general_stop.update(sampah_visual) 
    
    # B. Stopwords KHUSUS (Agar kata tidak "bocor" ke sentimen lain)
    
    # Netral: Hapus kata sentimen kuat & religius
    stop_netral = general_stop.copy()
    stop_netral.update([
        'pahala', 'dosa', 'haram', 'surga', 'neraka', 'ibadah',
        'syurga', 'masyaallah', 'ustadz', 'ustad', 'kill'
    ])

    # Negatif: Hapus kata positif & pujian
    stop_negatif = general_stop.copy()
    stop_negatif.update([
        'pahala', 'alhamdulillah', 'masyaallah', 'mantap', 'keren',
        'semangat', 'ustadz', 'ustad', 'terima', 'kasih'
    ])

    # Positif: Hapus kata negatif & teknis game kasar
    stop_positif = general_stop.copy()
    stop_positif.update([
        'gabut', 'sampah', 'beban', 'aneh', 'kalah', 'bocil',
        'haram', 'dosa', 'user', 'kill', 'kil', 'mati'
    ])
    
    return stop_netral, stop_negatif, stop_positif

# --- PREPROCESSING FUNCTIONS ---
def clean_for_model(text):
    """Preprocessing ringan khusus untuk input ke Model"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return " ".join(text.split())

def clean_for_visuals(text, stopword_set):
    """Preprocessing berat untuk WordCloud dengan Custom Stopwords"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    clean_words = []
    
    for w in words:
        # Normalisasi Slang dulu
        if w in norm_dict:
            w = norm_dict[w]
        
        # Filter menggunakan set stopword yang dipilih
        if w not in stopword_set:
            clean_words.append(w)
            
    return " ".join(clean_words)

# --- MAIN APP ---
def main():
    # Sidebar Info
    st.sidebar.title("Navigasi")
    menu = st.sidebar.radio("Pilih Menu:", ["Beranda", "Eksplorasi Data", "Prediksi Sentimen"])
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **Tentang Peneliti:**
        \nNama: Yazid Zaidan Alkhoir
        \nNIM: 402019611035
        \nProdi: Teknik Informatika
        \nUniversitas Darussalam Gontor
        """
    )

    model, vectorizer = load_assets()
    df = load_data()
    
    # Load custom stopwords
    stop_netral, stop_negatif, stop_positif = get_custom_stopwords()

    if menu == "Beranda":
        st.title("🎓 Sistem Analisis Sentimen Dakwah TikTok")
        st.subheader("Studi Kasus: Akun TikTok @Abiazkakiaa")
        st.markdown(
            """
            Aplikasi ini merupakan implementasi dari skripsi yang berjudul:
            **"Sentimen Analisis Dakwah Pada Media Sosial TikTok @Abiazkakiaa Menggunakan Metode SVM"**
            
            **Tujuan:**
            Menganalisis sentimen dan respon netizen terhadap konten dakwah yang disampaikan melalui media game Mobile Legends.
            
            **Metode:**
            - **Algoritma:** Support Vector Machine (SVM) kernel Linear.
            - **Feature Extraction:** TF-IDF.
            - **Preprocessing:** Hybrid (Regex + Stopword Removal + Normalisasi Slang).
            """
        )

    elif menu == "Eksplorasi Data":
        st.title("📊 Eksplorasi Data")
        
        if df is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Data Bersih", len(df))
            with col2:
                # Menghitung jumlah label terbanyak
                top_sentiment = df['sentiment'].mode()[0]
                st.metric("Sentimen Terbanyak", top_sentiment)

            st.write("### Sampel Data")
            st.dataframe(df[['clean_text', 'sentiment']].head(10))
            
            st.write("### Distribusi Sentimen")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(x='sentiment', data=df, palette='viridis', ax=ax)
            st.pyplot(fig)
            
            st.write("### WordCloud (Visualisasi Kata Sering Muncul)")
            sentiment_opt = st.selectbox("Pilih Kategori Sentimen untuk Visualisasi:", ["Positif", "Negatif", "Netral"])
            
            if st.button("Generate WordCloud"):
                # Tentukan stopword set berdasarkan pilihan
                if sentiment_opt == 'Positif':
                    selected_stopwords = stop_positif
                    colormap_style = 'Greens' # Warna hijau untuk positif
                elif sentiment_opt == 'Negatif':
                    selected_stopwords = stop_negatif
                    colormap_style = 'Reds'   # Warna merah untuk negatif
                else:
                    selected_stopwords = stop_netral
                    colormap_style = 'Blues'  # Warna biru untuk netral
                
                # Filter data berdasarkan sentimen
                filtered_df = df[df['sentiment'] == sentiment_opt]
                
                if not filtered_df.empty:
                    # Terapkan cleaning khusus visualisasi
                    text_data = filtered_df['clean_text'].apply(lambda x: clean_for_visuals(x, selected_stopwords))
                    text_combined = " ".join(text_data)
                    
                    if len(text_combined) > 0:
                        wc = WordCloud(
                            width=800, 
                            height=400, 
                            background_color='white', 
                            colormap=colormap_style,
                            stopwords=selected_stopwords # Backup safety
                        ).generate(text_combined)
                        
                        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                    else:
                        st.warning("Tidak ada kata yang cukup untuk ditampilkan setelah pembersihan stopwords.")
                else:
                    st.warning(f"Tidak ada data untuk sentimen {sentiment_opt}")

        else:
            st.error("File dataset (data_bersih.xlsx/csv) tidak ditemukan. Harap upload file dataset.")

    elif menu == "Prediksi Sentimen":
        st.title("🤖 Prediksi Sentimen Real-Time")
        
        if model is None or vectorizer is None:
            st.error("Model belum dimuat! Pastikan file 'svm_model.pkl' dan 'tfidf_vectorizer.pkl' ada di folder yang sama.")
        else:
            user_input = st.text_area("Masukkan Komentar TikTok:", placeholder="Contoh: Keren banget tadz cara dakwahnya...")
            
            if st.button("Analisis"):
                if user_input.strip() == "":
                    st.warning("Mohon masukkan teks terlebih dahulu.")
                else:
                    # 1. Preprocessing (untuk model hanya regex & lowercase sederhana)
                    processed_text = clean_for_model(user_input)
                    st.write(f"**Teks diproses:** `{processed_text}`")
                    
                    # 2. Vectorization
                    text_vector = vectorizer.transform([processed_text])
                    
                    # 3. Prediction
                    prediction = model.predict(text_vector)[0]
                    proba = model.predict_proba(text_vector)
                    confidence = np.max(proba) * 100
                    
                    # 4. Result Display
                    if prediction == 'Positif':
                        st.success(f"Hasil: **POSITIF** (Confidence: {confidence:.2f}%)")
                    elif prediction == 'Negatif':
                        st.error(f"Hasil: **NEGATIF** (Confidence: {confidence:.2f}%)")
                    else:
                        st.info(f"Hasil: **NETRAL** (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    main()