import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import random
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Analisis Sentimen TikTok - Skripsi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CACHE: STOPWORDS NLTK
# =========================
@st.cache_resource
def ensure_nltk_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    try:
        return pd.read_excel("data_bersih.xlsx")
    except Exception:
        try:
            return pd.read_csv("data_bersih.csv")
        except Exception:
            return None


# =========================
# (A) KAMUS NORMALISASI (Hybrid Preprocessing)
# =========================
norm_dict = {
    'ust': 'ustadz', 'ustd': 'ustadz', 'tad': 'ustadz', 'ml': 'mobile legends','ustad': 'ustadz',
    'mm': 'marksman', 'roam': 'roamer', 'tenk': 'tank', 'ks': 'nyampah', "asu": "anjing","gblk": "goblok",
    "gblg": "goblok","gblokk": "goblok", "toll": "tolol", "tolll": "tolol","tololl": "tolol","tololll": "tolol",
    "kampng": "kampung", "kampret": "kampret",'kocak': 'lucu', 'auto': 'langsung','real': 'nyata',
    'literally': 'benar-benar','fr': 'benar-benar', 'capek': 'lelah','malu2in': 'memalukan','idk': 'i dont know',
    'imo': 'in my opinion','imho': 'in my humble opinion','ikr': 'i know right','tbh': 'to be honest',
    'btw': 'by the way','omg': 'oh my god','wtf': 'kata kasar','wth': 'apa-apaan','lmao': 'tertawa',
    'lol': 'tertawa','rofl': 'tertawa','smh': 'geleng kepala','brb': 'segera kembali','afaik': 'sejauh yang saya tahu',
    'thx': 'terima kasih','ty': 'terima kasih','pls': 'tolong','plz': 'tolong','u': 'kamu','ur': 'punyamu','r': 'adalah','ya': 'iya',
    'yea': 'iya','yeah': 'iya','fuq': 'kata kasar','fck': 'kata kasar','fkn': 'kata kasar','fkng': 'kata kasar',
    'fckn': 'kata kasar','stfu': 'diam','bs': 'omong kosong','asf': 'sekali','mf': 'kata kasar','ffs': 'kata kasar',
    'gw': 'saya','gua': 'saya','gue': 'saya','loe': 'kamu','lu': 'kamu','elo': 'kamu','bro': 'saudara','sis': 'saudari',
    'sob': 'teman','cuan': 'untung','ngegas': 'marah','baper': 'terbawa perasaan','mager': 'malas gerak',
    'gabut': 'tidak ada kerjaan','nganggur': 'tidak bekerja','santuy': 'santai','sabi': 'bisa','wkwkw': 'haha',
    'anjss': 'kata kasar','tololl': 'tolol','anjg': 'kata kasar',"tai": "tai","tahi": "tai","pntd": "pantat",
    "pantek": "pantat", "push": "dorong","rank": "peringkat","mabar": "main bareng","ulti": "ultimate",
    "skill": "kemampuan","feed": "mati terus","feeder": "pemain mati terus","noob": "pemula","pro": "pemain hebat",
    "afk": "keluar dari permainan","lag": "lambat","ml": "mobile legends","hero": "pahlawan","match": "pertandingan",
    "epic": "peringkat epic","mythic": "peringkat mythic","gm": "peringkat grandmaster","lord": "boss besar",
    "minion": "prajurit kecil","buff": "peningkatan kekuatan","nerf": "penurunan kekuatan","damage": "kerusakan",
    "tank": "penahan serangan","mm": "marksman","mage": "penyihir","assassin": "pembunuh","fighter": "petarung",
    "support": "pendukung","jungler": "pemburu hutan","exp": "pengalaman","turret": "menara","base": "markas",
    "mid": "jalur tengah","bot": "jalur bawah","top": "jalur atas","draft": "pemilihan hero","ban": "larangan hero",
    "pick": "pilihan hero","gg": "bagus sekali","ez": "mudah","wp": "kerja bagus","op": "terlalu kuat","ga": "tidak",
    "gak": "tidak","gk": "tidak","nggak": "tidak","ngga": "tidak","aja": "saja","kok": "mengapa","dong": "",
    "nih": "ini","yaampun": "astaga","yaallah": "ya allah","astaga": "astaga","pls": "tolong","plis": "tolong",
    "sumpah": "sungguh","parah": "buruk sekali","kocak": "lucu",'by one': 'duel', 'mechanic': 'mekanik', 'op': 'kuat',
    'feed': 'beban','yg': 'yang', 'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'gk': 'tidak',
    'bgt': 'banget', 'sdh': 'sudah', 'kalo': 'kalau', 'tp': 'tapi',
    'jd': 'jadi', 'krn': 'karena', 'd': 'di', 'dr': 'dari', 'dgn': 'dengan',
    'utk': 'untuk', 'org': 'orang', 'sy': 'saya', 'gw': 'gue', 'lu': 'kamu',
    'tau': 'tahu', 'ni': 'ini', 'tu': 'itu', 'sm': 'sama', 'aja': 'saja',
    'emg': 'memang', 'anjir': 'anjing', 'njir': 'anjing', 'cok': 'cok',
    'wkwk': 'tertawa', 'bntu': 'bantu',"bgt": "banget",
    "bngt": "banget",  "lucu bgt": "lucu banget", "auto": "langsung", 'wkakaka': 'haha',
    'hehe': 'haha', 'huhu': 'sedih', 'huft': 'lelah', 'anj': 'anjing',
    'anjir': 'anjing', 'anjay': 'anjing', 'haha':'tertawa', 'anj': 'kata kasar',
    'anjir': 'kata kasar', 'anjay': 'kata kasar', 'njir': 'kata kasar', 'ngntd': 'kata kasar',
    'ngtnd': 'kata kasar', 'ngentd': 'kata kasar', 'mmk': 'kata kasar', 'ktl': 'kata kasar',
    'kontl': 'kata kasar', 'bangsd': 'kata kasar', 'bngsd': 'kata kasar', 'bngsat': 'kata kasar',
    'asu': 'kata kasar', 'gblk': 'goblok', 'gblg': 'goblok', 'gblokk': 'goblok', 'toll': 'tolol',
    'tolll': 'tolol', 'tololl': 'tolol', 'kampng': 'kampung', 'kampret': 'kata kasar', 'tai': 'kata kasar',
    "real": "nyata", "literally": "benar-benar", "fr": "benar-benar", "capek": "lelah", "malu2in": "memalukan",
    'bgt': 'banget', 'bngt': 'banget', 'bgd': 'banget', 'bngd': 'banget', 'btw': 'ngomong-ngomong',
    'wkwk': 'haha', 'wkwwk': 'haha', 'wkwkwk': 'haha', 'tahi': 'kata kasar', 'pntd': 'kata kasar',
    'pantek': 'kata kasar', 'cok': 'kata kasar', 'asw': 'kata kasar','push': 'dorong', 'rank': 'peringkat',
    'mabar': 'main bareng', 'ulti': 'ultimate', 'skill': 'kemampuan', 'feed': 'mati terus', 'feeder': 'pemain mati terus',
    'noob': 'pemula', 'pro': 'pemain hebat', 'afk': 'keluar dari permainan', 'lag': 'lambat', 'ml': 'mobile legends',
    'hero': 'pahlawan', 'match': 'pertandingan', 'epic': 'peringkat epic','mythic': 'peringkat mythic',
    'gm': 'peringkat grandmaster', 'lord': 'boss besar', 'minion': 'prajurit kecil', 'buff': 'peningkatan kekuatan',
    'nerf': 'penurunan kekuatan', 'damage': 'kerusakan', 'tank': 'penahan serangan', 'mm': 'marksman',
    'mage': 'penyihir', 'assassin': 'pembunuh', 'fighter': 'petarung', 'support': 'pendukung', 'jungler': 'pemburu hutan',
    'exp': 'pengalaman', 'turret': 'menara', 'base': 'markas', 'mid': 'jalur tengah', 'bot': 'jalur bawah',
    'top': 'jalur atas', 'draft': 'pemilihan hero', 'ban': 'larangan hero', 'pick': 'pilihan hero',
    'gg': 'bagus sekali', 'ez': 'mudah', 'wp': 'kerja bagus', 'op': 'terlalu kuat','ga': 'tidak',
    'gak': 'tidak', 'gk': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak', 'aja': 'saja', 'kok': 'mengapa',
    'dong': '', 'nih': 'ini', 'yaampun': 'astaga', 'yaallah': 'ya allah', 'astaga': 'astaga', 'pls': 'tolong',
    'plis': 'tolong', 'sumpah': 'sungguh', 'parah': 'buruk sekali', "bgd": "banget", "bngd": "banget",
    "btw": "ngomong-ngomong", "wkwk": "haha", "wkwwk": "haha", "wkwkwk": "haha", "wkakaka": "haha",
    "hehe": "haha", "huhu": "sedih", "huft": "lelah", "anj": "anjing", "anjir": "anjing", "anjay": "anjing",
    "njir": "anjing", "ngntd": "ngentot", "ngtnd": "ngentot", "ngentd": "ngentot", "mmk": "memek",
    "ktl": "kontol", "kontl": "kontol", "bangsd": "bangsat", "bngsd": "bangsat", "bngsat": "bangsat",
}

# =========================
# (B) Sampah Visual WordCloud
# =========================
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
    'santuy', 'sabi', 'wkwkw', 'anjss', 'tololl', 'anjg', 'pa', 'sia', 'ko', 'klau''nya', 'sih', 'kok', 'mah', 'deh', 'dong', 'kah', 'ni', 'tu', 'tuh',
    'yah', 'wow', 'dan', 'yang', 'di', 'itu', 'ini', 'ke', 'dari',
    'yg', 'udah', 'sdh', 'dah', 'dr', 'kalo', 'klu', 'utk', 'ya', 'yu', 'y',
    'juga', 'jg', 'km', 'ak', 'aku', 'kami', 'kita', 'orang', 'biar',
    'gitu', 'pake', 'karna', 'dpt', 'gini', 'kon', 'ku', 'jdi', 'la', 'kn', 'kan',
    'game', 'games', 'gim', 'play', 'player', 'montoon', 'moonton',
}

# =========================
# STOPWORDS SETUP
# =========================
@st.cache_resource
def get_custom_stopwords(sampah_visual: set):
    ensure_nltk_stopwords()
    stop_factory = set(stopwords.words("indonesian"))

    general_stop = set(STOPWORDS)
    general_stop.update(stop_factory)
    general_stop.update(sampah_visual)

    # Stopwords khusus
    stop_netral = general_stop.copy()
    stop_netral.update(["pahala","dosa","haram","surga","neraka","ibadah","syurga","masyaallah","ustadz","ustad","kill"])

    stop_negatif = general_stop.copy()
    stop_negatif.update(["pahala","alhamdulillah","masyaallah","mantap","keren","semangat","ustadz","ustad","terima","kasih"])

    stop_positif = general_stop.copy()
    stop_positif.update(["gabut","sampah","beban","aneh","kalah","bocil","haram","dosa","user","kill","kil","mati"])

    return stop_netral, stop_negatif, stop_positif


# =========================
# NORMALISASI LABEL SENTIMEN -> FIX JADI 3 KELAS
# =========================
def normalize_sentiment_label(x: str) -> str:
    if pd.isna(x):
        return "Netral"
    s = str(x).strip().lower()

    # bersihin karakter aneh
    s = re.sub(r"[^a-z0-9\s_-]", "", s).strip()

    mapping = {
        "positif": "Positif", "positive": "Positif", "pos": "Positif", "+": "Positif", "1": "Positif",
        "negatif": "Negatif", "negative": "Negatif", "neg": "Negatif", "-": "Negatif", "0": "Negatif",
        "netral": "Netral", "neutral": "Netral", "neu": "Netral", "2": "Netral"
    }

    for k, v in mapping.items():
        if s == k:
            return v

    if "pos" in s:
        return "Positif"
    if "neg" in s:
        return "Negatif"
    return "Netral"


# =========================
# PREPROCESS
# =========================
def clean_for_visuals(text, stopword_set):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()

    clean_words = []
    for w in words:
        if w in norm_dict:
            w = norm_dict[w]
        if w not in stopword_set:
            clean_words.append(w)

    return " ".join(clean_words)


# =========================
# UTIL: FIGURE -> PNG BYTES
# =========================
def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


# =========================
# VISUALISASI: BAR + PIE (Matplotlib)
# =========================
def make_bar_chart(counts):
    # Buat kanvas
    fig, ax = plt.subplots(figsize=(6, 4))

    # Definisi warna per sentimen (Sesuai Konteks)
    # Merah = Negatif, Biru = Netral, Hijau = Positif
    colors = []
    for label in counts.index:
        if label == "Positif":
            colors.append("#2ecc71")  # Hijau segar
        elif label == "Negatif":
            colors.append("#e74c3c")  # Merah
        else:
            colors.append("#3498db")  # Biru (Netral)

    # Plot dengan parameter color=colors
    bars = ax.bar(counts.index, counts.values, color=colors)

    # Tambahkan label angka di atas setiap bar agar informatif
    ax.bar_label(bars, padding=3)

    # Label dan Judul
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah")
    ax.set_title("Distribusi Sentimen (Bar Plot)")

    # Agar layout pas
    fig.tight_layout()
    return fig


def make_pie_chart(counts):
    fig, ax = plt.subplots(figsize=(6, 4))

    # Samakan warna Pie Chart dengan Bar Chart agar konsisten
    colors_map = {"Positif": "#2ecc71", "Negatif": "#e74c3c", "Netral": "#3498db"}
    pie_colors = [colors_map.get(label, "grey") for label in counts.index]

    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=pie_colors)
    ax.set_title("Distribusi Sentimen (Pie Chart)")
    ax.axis("equal")
    fig.tight_layout()
    return fig


# =========================
# WORDCLOUD
# =========================
def color_func_positif(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(120, 100%%, %d%%)" % random.randint(25, 45)  # Hijau gelap

def color_func_netral(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(210, 100%%, %d%%)" % random.randint(25, 45)  # Biru gelap

def color_func_negatif(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 100%%, %d%%)" % random.randint(25, 45)    # Merah gelap


def make_wordcloud(text, stopwords_set, color_func):
    wc = WordCloud(
        width=1100,
        height=520,
        background_color="white",
        stopwords=stopwords_set,
        collocations=False,
        max_words=200
    ).generate(text)

    wc = wc.recolor(color_func=color_func)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout()
    return fig


# =========================
# MAIN APP
# =========================
def main():
    st.sidebar.title("Navigasi")
    menu = st.sidebar.radio("Pilih Menu:", ["Beranda", "Eksplorasi Data","Evaluasi","Visualisasi Sentimen", "WordCloud"])

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Tentang Peneliti:**\n"
        "\nNama: Yazid Zaidan Alkhoir"
        "\nNIM: 402019611035"
        "\nProdi: Teknik Informatika"
        "\nUniversitas Darussalam Gontor"
    )

    df = load_data()
    if df is None:
        st.error("Dataset tidak ditemukan. Pastikan file `data_bersih.xlsx` / `data_bersih.csv` ada di folder yang sama dengan app.py")
        return

    if "sentiment" not in df.columns or "clean_text" not in df.columns:
        st.error("Kolom wajib tidak ditemukan. Dataset harus punya kolom: `clean_text` dan `sentiment`.")
        return

    # rapikan data
    df = df.dropna(subset=["clean_text", "sentiment"]).copy()
    df["sentiment"] = df["sentiment"].apply(normalize_sentiment_label)

    # FIX: pastikan cuma 3 kelas yang kepake
    allowed = {"Negatif", "Netral", "Positif"}
    df = df[df["sentiment"].isin(allowed)].copy()

    # stopwords
    stop_netral, stop_negatif, stop_positif = get_custom_stopwords(sampah_visual)

    # distribusi
    order = ["Negatif", "Netral", "Positif"]
    counts = df["sentiment"].value_counts().reindex(order).fillna(0).astype(int)

    if menu == "Beranda":
        st.title("🎓 Sistem Analisis Sentimen Dakwah TikTok")
        st.subheader("Studi Kasus: Akun TikTok @AbiAzkakiaa")
        st.markdown('Aplikasi ini merupakan implementasi dari skripsi:\n\n'
                    '**"Analisis Sentimen Dakwah Pada Media Sosial TikTok @Abiazkakiaa Menggunakan Metode SVM"**')

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Data Bersih", len(df))
        c2.metric("Jumlah Kelas", 3)
        c3.metric("Sentimen Terbanyak", df["sentiment"].mode()[0])

    elif menu == "Eksplorasi Data":
        st.title("📊 Eksplorasi Data")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Data Bersih", len(df))
        c2.metric("Sentimen Terbanyak", df["sentiment"].mode()[0])
        c3.metric("Jumlah Kelas", 3)

        st.markdown("### Filter Data per Sentimen")
        pilihan = st.selectbox("Pilih data yang ingin ditampilkan:", ["Semua", "Positif", "Netral", "Negatif"])

        if pilihan == "Semua":
            df_show = df
        else:
            df_show = df[df["sentiment"] == pilihan]

        st.write(f"Jumlah data ditampilkan: **{len(df_show)}**")
        st.dataframe(df_show[["clean_text", "sentiment"]].head(50), use_container_width=True)

    elif menu == "Evaluasi":
        st.title("📊 Evaluasi Model SVM")

        FILE_PATH = "svm_metrics.xlsx"

        try:
            # =========================
            # LOAD DATA
            # =========================
            report_df = pd.read_excel(FILE_PATH, sheet_name="classification_report")
            cm_df = pd.read_excel(FILE_PATH, sheet_name="confusion_matrix")

            # =========================
            # AKURASI
            # =========================
            st.subheader("✅ Akurasi Model (Data Uji)")

            # 1) Coba ambil dari classification_report (baris 'accuracy')
            acc = None
            try:
                acc_row = report_df[report_df["label"] == "accuracy"]
                if not acc_row.empty:
                    # biasanya akurasi tersimpan di kolom 'precision' (karena output_dict sklearn)
                    for col in ["accuracy", "precision", "f1-score"]:
                        if col in acc_row.columns:
                            val = acc_row.iloc[0][col]
                            if pd.notna(val):
                                acc = float(val)
                                break
            except Exception:
                acc = None

            # 2) Kalau tidak ketemu, hitung dari confusion matrix
            if acc is None:
                cm_tmp = cm_df.copy()
                if "actual" in cm_tmp.columns:
                    cm_tmp = cm_tmp.set_index("actual")
                cm_tmp = cm_tmp.apply(pd.to_numeric, errors="coerce").fillna(0)

                total = cm_tmp.values.sum()
                correct = sum(cm_tmp.iloc[i, i] for i in range(min(cm_tmp.shape)))
                acc = (correct / total) if total > 0 else 0.0

            # Tampilkan
            st.metric("Accuracy", f"{acc * 100:.2f}%")

            # =========================
            # CLASSIFICATION REPORT
            # =========================
            st.subheader("📄 Classification Report")
            st.dataframe(report_df, use_container_width=True)

            # =========================
            # BAR CHART PR / RE / F1
            # =========================
            st.subheader("📊 Precision, Recall, F1-score per Kelas")

            # buang baris ringkasan
            report_plot = report_df[
                ~report_df["label"].isin(["accuracy", "macro avg", "weighted avg"])
            ]

            fig, ax = plt.subplots(figsize=(8, 5))
            x = range(len(report_plot))

            ax.bar(x, report_plot["precision"], width=0.25, label="Precision")
            ax.bar([i + 0.25 for i in x], report_plot["recall"], width=0.25, label="Recall")
            ax.bar([i + 0.50 for i in x], report_plot["f1-score"], width=0.25, label="F1-score")

            ax.set_xticks([i + 0.25 for i in x])
            ax.set_xticklabels(report_plot["label"])
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score")
            ax.legend()

            st.pyplot(fig)

            # =========================
            # CONFUSION MATRIX
            # =========================
            st.subheader("🧩 Confusion Matrix")

            # rapikan dataframe
            if "actual" in cm_df.columns:
                cm_df = cm_df.set_index("actual")

            cm_df = cm_df.apply(pd.to_numeric)

            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            im = ax_cm.imshow(cm_df.values)

            ax_cm.set_xticks(range(len(cm_df.columns)))
            ax_cm.set_yticks(range(len(cm_df.index)))
            ax_cm.set_xticklabels(cm_df.columns)
            ax_cm.set_yticklabels(cm_df.index)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")

            # angka di sel
            for i in range(cm_df.shape[0]):
                for j in range(cm_df.shape[1]):
                    ax_cm.text(j, i, cm_df.iloc[i, j],
                               ha="center", va="center")

            fig_cm.colorbar(im, ax=ax_cm)
            st.pyplot(fig_cm)

        except FileNotFoundError:
            st.error("❌ File svm_metrics.xlsx tidak ditemukan.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

    elif menu == "Visualisasi Sentimen":
        st.title("📈 Visualisasi Distribusi Sentimen")

        left, right = st.columns(2)

        with left:
            st.subheader("Bar Plot")
            fig_bar = make_bar_chart(counts)
            st.image(fig_to_png_bytes(fig_bar), use_container_width=True)

        with right:
            st.subheader("Pie Chart")
            # Fungsi Pie Chart juga saya sesuaikan warnanya agar konsisten
            fig_pie = make_pie_chart(counts)
            st.image(fig_to_png_bytes(fig_pie), use_container_width=True)

        st.markdown("### Ringkasan Distribusi")
        ringkasan = pd.DataFrame({
            "No": [1, 2, 3],
            "Sentimen": counts.index.tolist(),
            "Jumlah": counts.values.tolist()
        })
        st.dataframe(ringkasan, use_container_width=True)

    elif menu == "WordCloud":
        st.title("☁️ WordCloud Kata Dominan")

        sentiment_opt = st.selectbox("Pilih Kategori Sentimen:", ["Positif", "Netral", "Negatif"])

        if sentiment_opt == "Positif":
            selected_stopwords = stop_positif
            color_func = color_func_positif
        elif sentiment_opt == "Negatif":
            selected_stopwords = stop_negatif
            color_func = color_func_negatif
        else:
            selected_stopwords = stop_netral
            color_func = color_func_netral

        if st.button("Generate WordCloud"):
            filtered_df = df[df["sentiment"] == sentiment_opt]
            if filtered_df.empty:
                st.warning(f"Tidak ada data untuk sentimen {sentiment_opt}.")
                return

            text_data = filtered_df["clean_text"].apply(lambda x: clean_for_visuals(x, selected_stopwords))
            text_combined = " ".join(text_data).strip()

            if not text_combined:
                st.warning("Teks kosong setelah preprocessing & stopwords.")
                return

            fig_wc = make_wordcloud(text_combined, selected_stopwords, color_func=color_func)
            st.image(fig_to_png_bytes(fig_wc), use_container_width=True)

            with st.expander("Lihat contoh data (20 baris)"):
                st.dataframe(filtered_df[["clean_text", "sentiment"]].head(20), use_container_width=True)


if __name__ == "__main__":
    main()
