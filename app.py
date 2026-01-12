import streamlit as st
import pandas as pd
import pickle
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# FUNCTION - LOAD PKL
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# FUNCTION - CONFUSION MATRIX
def plot_confusion_matrix_small(cm, labels, title):
    """
    Menampilkan confusion matrix kecil & rapi (untuk Streamlit)
    """
    fig, ax = plt.subplots(figsize=(2.2, 2.0))

    ax.imshow(cm, cmap="Blues")

    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=False)

from wordcloud import WordCloud
from collections import defaultdict
from preprocessing import preprocessing_pipeline
from dld_correction import dld_correct_text
from aspek_detector import detect_aspek
from aspek_detector import detect_aspek, aspek_keywords
from streamlit_option_menu import option_menu


# PAGE CONFIG
st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    layout="wide"
)

# CSS
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #0B2C4D;
}
section[data-testid="stSidebar"] * {
    color: white;
}
.stButton>button {
    background-color: #0B2C4D;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# FUNGSI LOAD MODEL
@st.cache_resource
def load_model(aspek, model_key):

    aspek_map = {
        "Daya Tarik": "daya_tarik",
        "Aksesibilitas": "aksesibilitas",
        "Amenitas": "amenitas",
        "Pelayanan Tambahan": "pelayanan_tambahan"
    }

    model_map = {
        "Baseline + DLD": (
            "models/svm_dld_baseline",
            f"{aspek_map[aspek]}_dld_baseline.pkl",
            True
        ),
        "Unigram + DLD": (
            "models/svm_dld_unigram",
            f"{aspek_map[aspek]}_dld_unigram.pkl",
            True
        ),
        "Unigram+Bigram + DLD": (
            "models/svm_dld_unigram_bigram",
            f"{aspek_map[aspek]}_dld_unigram_bigram.pkl",
            True
        ),
        "Unigram+Bigram+Trigram + DLD": (
            "models/svm_dld_unigram_bigram_trigram",
            f"{aspek_map[aspek]}_dld_unigram_bigram_trigram.pkl",
            True
        ),
        "Baseline (Tanpa DLD)": (
            "models/svm_no_dld_baseline",
            f"{aspek_map[aspek]}_no_dld_baseline.pkl",
            False
        ),
        "Unigram (Tanpa DLD)": (
            "models/svm_no_dld_unigram",
            f"{aspek_map[aspek]}_no_dld_unigram.pkl",
            False
        ),
        "Unigram+Bigram (Tanpa DLD)": (
            "models/svm_no_dld_unigram_bigram",
            f"{aspek_map[aspek]}_no_dld_unigram_bigram.pkl",
            False
        ),
        "Unigram+Bigram+Trigram (Tanpa DLD)": (
            "models/svm_no_dld_unigram_bigram_trigram",
            f"{aspek_map[aspek]}_no_dld_unigram_bigram_trigram.pkl",
            False
        )
    }

    folder, filename, use_dld = model_map[model_key]
    path = os.path.join(folder, filename)

    if not os.path.exists(path):
        st.error(f"Model tidak ditemukan: {path}")
        st.stop()

    with open(path, "rb") as f:
        obj = pickle.load(f)

    return obj["model"], obj["vectorizer"], use_dld


# FUNGSI KEYWORD
def get_aspek_keywords(text, aspek):
    text = text.lower()
    found = []

    for kw in aspek_keywords.get(aspek, []):
        if kw in text:
            found.append(kw)

    return ", ".join(found) if found else "-"

# FUNGSI GRAFIK
def annotate_bar(ax, x, height, fold):
    ax.text(
        x,
        height + 0.005,
        f"F{fold}\n{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=8
    )



# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    # =========================
    # LOGO (CENTER & BESAR)
    # =========================
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("logo.png", width=170)

    st.markdown(
        """
        <p style='
            text-align: center;
            color: #E5E7EB;
            font-size: 13px;
            margin-top: -10px;
            margin-bottom: 20px;
        '>
        Analisis Sentimen Ulasan Wisata<br>
        <b>Kabupaten Bangkalan</b>
        </p>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # MENU SIDEBAR
    # =========================
    menu = option_menu(
        "MENU UTAMA",
        [
            "Analisis",
            "Data Wisata",
            "Teks Augmentasi",
            "Preprocessing",
            "Hasil DLD",
            "Fitur N-Gram",
            "Matriks TF-IDF",
            "Uji Coba N-Gram Tunggal",
            "Uji Coba Rentang N-Gram",
            "Perbandingan Penggunaan DLD",
        ],
        icons=[
            "bar-chart-line",   # Analisis (BERANDA)
            "database",
            "shuffle",
            "gear",
            "bar-chart",
            "cloud",
            "diagram-3",
            "layers",
            "graph-up"
        ],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {
                "background-color": "#0E2A47",
                "padding": "8px"
            },
            "menu-title": {
                "color": "#FFFFFF",
                "font-weight": "700",
                "font-size": "16px",
                "margin-bottom": "10px"
            },
            "icon": {
                "color": "#FFFFFF",
                "font-size": "16px"
            },
            "nav-link": {
                "color": "#E5E7EB",
                "font-size": "14px",
                "padding": "10px 12px",
                "border-radius": "8px"
            },
            "nav-link-selected": {
                "background-color": "#2563EB",
                "color": "#FFFFFF",
                "font-weight": "600"
            }
        }
    )


# =====================================================
# BERANDA
# =====================================================
if menu == "Analisis":

    st.title("Dashboard Analisis Sentimen")
    st.caption(
        "Analisis Sentimen Multi-Aspek Ulasan Wisata Pantai "
        "di Kabupaten Bangkalan"
    )

    # PILIH MODEL
    st.subheader("⚙️ Pilih Model Analisis")

    selected_model = st.selectbox(
        "Gunakan skenario model:",
        [
            "Baseline + DLD",
            "Unigram + DLD",
            "Unigram+Bigram + DLD",
            "Unigram+Bigram+Trigram + DLD",
            "Baseline (Tanpa DLD)",
            "Unigram (Tanpa DLD)",
            "Unigram+Bigram (Tanpa DLD)",
            "Unigram+Bigram+Trigram (Tanpa DLD)"
        ]
    )

    tab_manual, tab_csv = st.tabs([
        "Input Teks Manual",
        "Analisis Batch (Upload CSV)"
    ])

    # TAB MANUAL
    with tab_manual:

        input_text = st.text_area(
            "Masukkan 1 ulasan wisata:",
            placeholder="Contoh: Pantainya indah tapi akses jalannya rusak..."
        )

        if st.button("🔍 Analisis Sentimen", key="manual"):

            if input_text.strip() == "":
                st.warning("Ulasan tidak boleh kosong")
                st.stop()

            # PREPROCESSING
            prep = preprocessing_pipeline(input_text)
            aspek_detected = detect_aspek(prep["cleaning"])

            rows = []
            preprocess_rows = []

            if aspek_detected != "Tidak Teridentifikasi":
                aspek_list = [a.strip() for a in aspek_detected.split(",")]

                # PROSES PREPROCESSING
                if "DLD" in selected_model:
                    preprocess_rows = [
                        {"Tahap": "Case Folding", "Hasil": prep["case_folding"]},
                        {"Tahap": "Cleaning", "Hasil": prep["cleaning"]},
                        {"Tahap": "Tokenizing", "Hasil": ", ".join(prep["tokenizing"])},
                        {"Tahap": "Normalisasi", "Hasil": ", ".join(prep["normalisasi"])},
                        {"Tahap": "Stopword Removal", "Hasil": ", ".join(prep["stopword_removal"])},
                        {"Tahap": "Stemming", "Hasil": " ".join(prep["stemming"])},
                        {"Tahap": "Teks Sebelum DLD", "Hasil": prep["final_text"]},
                    ]
                else:
                    preprocess_rows = [
                        {"Tahap": "Case Folding", "Hasil": prep["case_folding"]},
                        {"Tahap": "Cleaning", "Hasil": prep["cleaning"]},
                        {"Tahap": "Tokenizing", "Hasil": ", ".join(prep["tokenizing"])},
                        {"Tahap": "Normalisasi", "Hasil": ", ".join(prep["normalisasi"])},
                        {"Tahap": "Stopword Removal", "Hasil": ", ".join(prep["stopword_removal"])},
                        {"Tahap": "Stemming", "Hasil": " ".join(prep["stemming"])},
                        {"Tahap": "Teks Final", "Hasil": prep["final_text"]},
                    ]

                for aspek in aspek_list:

                    model, vectorizer, use_dld = load_model(
                        aspek, selected_model
                    )

                    # ALUR TEKS KE MODEL
                    if use_dld:
                        text_model = dld_correct_text(prep["final_text"])

                        # Tambahkan hasil DLD ke tabel
                        if not any(r["Tahap"] == "Teks Setelah DLD" for r in preprocess_rows):
                            preprocess_rows.append({
                                "Tahap": "Teks Setelah DLD",
                                "Hasil": text_model
                            })
                    else:
                        text_model = prep["final_text"]

                    X = vectorizer.transform([text_model])
                    pred_label = model.predict(X)[0]

                    if pred_label == "Positif":
                        pred = 1
                    elif pred_label == "Negatif":
                        pred = -1
                    else:
                        pred = 0

                    if pred_label == "Positif":
                        sentimen_text = "Positif"
                    elif pred_label == "Negatif":
                        sentimen_text = "Negatif"
                    else:
                        sentimen_text = "Tidak Memuat"

                    rows.append({
                        "Aspek": aspek,
                        "Teks Asli": input_text,
                        "Kata Kunci Aspek": get_aspek_keywords(prep["cleaning"], aspek),
                        "Teks Final": text_model,
                        "Sentimen": sentimen_text
                    })


            # TAMPILKAN PREPROCESSING
            st.subheader("🧪 Proses Preprocessing")
            st.dataframe(
                pd.DataFrame(preprocess_rows),
                use_container_width=True
            )

            # TAMPILKAN HASIL ANALISIS
            st.subheader("📊 Hasil Analisis")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


    # TAB CSV
    with tab_csv:

        file = st.file_uploader(
            "Upload file CSV (kolom: ulasan)",
            type=["csv"]
        )

        if file is not None:
            df = pd.read_csv(file)

            if "ulasan" not in df.columns:
                st.error("CSV harus memiliki kolom bernama 'ulasan'")
                st.stop()

            hasil_all = []

            for text in df["ulasan"]:

                prep = preprocessing_pipeline(text)
                aspek_detected = detect_aspek(prep["cleaning"])

                if aspek_detected != "Tidak Teridentifikasi":
                    aspek_list = [a.strip() for a in aspek_detected.split(",")]

                    for aspek in aspek_list:

                        model, vectorizer, use_dld = load_model(
                            aspek, selected_model
                        )

                        if use_dld:
                            text_model = dld_correct_text(prep["final_text"])
                        else:
                            text_model = prep["final_text"]

                        X = vectorizer.transform([text_model])
                        pred_label = model.predict(X)[0]

                        if pred_label == "Positif":
                            pred = 1
                        elif pred_label == "Negatif":
                            pred = -1
                        else:
                            pred = 0

                        if pred_label == "Positif":
                            sentimen_text = "Positif"
                        elif pred_label == "Negatif":
                            sentimen_text = "Negatif"
                        else:
                            sentimen_text = "Tidak Memuat"

                        hasil_all.append({
                            "Ulasan": text,
                            "Aspek": aspek,
                            "Teks Asli": input_text,
                            "Kata Kunci Aspek": get_aspek_keywords(prep["cleaning"], aspek),
                            "Teks Final": text_model,
                            "Sentimen": sentimen_text
                        })


            st.subheader("📊 Hasil Analisis Batch")
            st.dataframe(pd.DataFrame(hasil_all), use_container_width=True)


# =====================================================
# DATA WISATA
# =====================================================
elif menu == "Data Wisata":
    st.title("📁 Data Wisata")
    df = pd.read_csv("data_wisata_baru.csv")
    st.write(f"Jumlah data: {len(df)}")
    st.dataframe(df)

# =====================================================
# TEKS AUGMENTASI
# =====================================================
elif menu == "Teks Augmentasi":
    import os
    import pandas as pd

    st.title("🔁 Hasil Teks Augmentasi")
    st.caption("Detail hasil augmentasi data")

    BASE_PATH = "visualisasi/hasil_augmentasi"

    mode = st.radio(
        "Mode Tampilan:",
        ["Semua Data", "Per Aspek"],
        horizontal=True
    )

    # =================================================
    # MODE 1 — SEMUA DATA
    # =================================================
    if mode == "Semua Data":

        st.subheader("📌 Semua Data Augmentasi")

        file_all = os.path.join(BASE_PATH, "augmentasi_all.pkl")

        if not os.path.exists(file_all):
            st.error("❌ File augmentasi_all.pkl tidak ditemukan")
            st.stop()

        data = load_pkl(file_all)

        df_all = data["df_all_train"].copy()
        df_aug = data["df_all_aug"].copy()
        stats  = data["stats"]

        # ===== RINGKASAN =====
        st.markdown("### 📊 Ringkasan Augmentasi")
        st.dataframe(stats, use_container_width=True)

        # ===== SEMUA DATA TRAIN =====
        st.markdown("### 📝 Seluruh Data Train (Setelah Augmentasi)")
        st.dataframe(
            df_all.rename(columns={"Ulasan": "Teks"})[
                ["Teks", "Sentimen"]
            ],
            use_container_width=True
        )

        # ===== DATA HASIL AUGMENTASI SAJA =====
        if not df_aug.empty:
            st.markdown("### 🧬 Data Hasil Augmentasi (Teks Baru)")
            st.dataframe(
                df_aug.rename(columns={"Ulasan": "Teks"})[
                    ["Teks", "Sentimen"]
                ],
                use_container_width=True
            )

    # =================================================
    # MODE 2 — PER ASPEK & PER FOLD
    # =================================================
    else:
        st.subheader("📌 Augmentasi per Aspek & Fold")

        aspek_list = sorted([
            d for d in os.listdir(BASE_PATH)
            if os.path.isdir(os.path.join(BASE_PATH, d))
        ])

        aspek = st.selectbox("Pilih Aspek:", aspek_list)
        aspek_path = os.path.join(BASE_PATH, aspek)

        fold_files = sorted([
            f for f in os.listdir(aspek_path)
            if f.endswith(".pkl")
        ])

        fold = st.selectbox("Pilih Fold:", fold_files)

        data = load_pkl(os.path.join(aspek_path, fold))

        df_train = data["df_train"].copy()
        df_aug   = data["df_aug_only"].copy()
        stat     = data["statistik"]

        # ===== METRIC =====
        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Total Train", len(df_train))
        col2.metric("🧬 Augmentasi Baru", len(df_aug))
        col3.metric("🗑️ Duplikat Dihapus", stat["Dup_Removed"])

        st.markdown(
            f"**Aspek:** {data['aspek']} | "
            f"**Fold:** {data['fold']}"
        )

        # ===== STATISTIK DETAIL (TABEL) =====
        st.markdown("### 📊 Statistik Fold")

        df_stat = (
            pd.DataFrame(stat.items(), columns=["Keterangan", "Nilai"])
        )

        st.dataframe(
            df_stat,
            use_container_width=True,
            hide_index=True
        )


        # ===== DATA TRAIN =====
        st.markdown("### 📝 Data Train (Setelah Augmentasi)")
        st.dataframe(
            df_train.rename(columns={"Ulasan": "Teks"})[
                ["Teks", "Sentimen"]
            ],
            use_container_width=True
        )

        # ===== DATA AUGMENTASI SAJA =====
        if not df_aug.empty:
            st.markdown("### 🧬 Teks Hasil Augmentasi Saja")
            st.dataframe(
                df_aug.rename(columns={"Ulasan": "Teks"})[
                    ["Teks", "Sentimen"]
                ],
                use_container_width=True
            )



# =====================================================
# PREPROCESSING (SEMUA TAHAP) — STREAMLIT
# =====================================================
elif menu == "Preprocessing":

    import os
    import pandas as pd
    import streamlit as st

    st.title("⚙️ Preprocessing Teks")
    st.caption("Hasil preprocessing per aspek, per fold, dan per jenis data")

    BASE_DIR = "visualisasi/hasil_preprocessing"

    # =====================================================
    # KONFIGURASI FILE PREPROCESSING
    # =====================================================
    tahap_config = {
        "Case Folding": "case_folding.pkl",
        "Cleaning": "cleaning.pkl",
        "Tokenisasi": "tokenisasi.pkl",
        "Normalisasi Slang": "normalisasi.pkl",
        "Stopword Removal": "stopword.pkl",
        "Stemming": "stemming.pkl"
    }

    tahap = st.selectbox(
        "Pilih Tahapan Preprocessing",
        list(tahap_config.keys())
    )

    file_path = os.path.join(BASE_DIR, tahap_config[tahap])

    if not os.path.exists(file_path):
        st.error("❌ File preprocessing tidak ditemukan")
        st.stop()

    # =====================================================
    # LOAD PKL
    # =====================================================
    obj = load_pkl(file_path)

    data = obj["data"]     # aspek → fold → train/test
    stats = obj["stats"]   # DataFrame statistik token/kata

    aspek = st.selectbox(
        "Pilih Aspek",
        sorted(data.keys())
    )

    fold = st.selectbox(
        "Pilih Fold",
        sorted(data[aspek].keys())
    )

    jenis = st.radio(
        "Pilih Jenis Data",
        ["train", "test"],
        horizontal=True
    )

    df = data[aspek][fold][jenis]

    if df.empty:
        st.warning("Data kosong")
        st.stop()

    # =====================================================
    # AMBIL STATISTIK TOKEN / KATA (TRAIN & TEST)
    # =====================================================
    df_stat = stats[
        (stats["Aspek"] == aspek) &
        (stats["Fold"] == fold)
    ]

    if not df_stat.empty:

        cols = df_stat.columns.tolist()

        def find_col(keyword):
            for c in cols:
                if keyword.lower() in c.lower():
                    return c
            return None

        # Aman untuk semua tahap (kata / token)
        train_before_col = (
            find_col("Train_Kata_Sebelum") or
            find_col("Train_Token_Sebelum")
        )
        train_after_col = (
            find_col("Train_Kata_Sesudah") or
            find_col("Train_Token_Sesudah")
        )
        test_before_col = (
            find_col("Test_Kata_Sebelum") or
            find_col("Test_Token_Sebelum")
        )
        test_after_col = (
            find_col("Test_Kata_Sesudah") or
            find_col("Test_Token_Sesudah")
        )

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "📘 TRAIN (Sebelum → Sesudah)",
                f"{int(df_stat[train_before_col].iloc[0])} → "
                f"{int(df_stat[train_after_col].iloc[0])}"
            )

        with c2:
            st.metric(
                "📕 TEST (Sebelum → Sesudah)",
                f"{int(df_stat[test_before_col].iloc[0])} → "
                f"{int(df_stat[test_after_col].iloc[0])}"
            )

    # =====================================================
    # INFORMASI KONTEKS
    # =====================================================
    st.markdown(
        f"""
        **Tahap:** {tahap}  
        **Aspek:** {aspek}  
        **Fold:** {fold}  
        **Jenis Data:** {jenis.upper()}
        """
    )

    # =====================================================
    # AMBIL KOLOM TEKS (HASIL TERAKHIR)
    # =====================================================
    kolom_teks = [
        c for c in df.columns
        if c.lower().startswith("ulasan")
    ]

    if len(kolom_teks) == 0:
        st.error("Kolom teks tidak ditemukan")
        st.stop()

    # Ambil hasil preprocessing (kolom terakhir)
    kolom_final = kolom_teks[-1]

    tampil = df[[kolom_final]].copy()
    tampil.columns = ["Teks"]

    st.dataframe(
        tampil,
        use_container_width=True
    )

    # =====================================================
    # OPSIONAL: TAMPILKAN SEBELUM → SESUDAH
    # =====================================================
    if len(kolom_teks) >= 2:
        with st.expander("🔍 Lihat Sebelum → Sesudah"):
            st.dataframe(
                df[[kolom_teks[0], kolom_teks[-1]]]
                .rename(columns={
                    kolom_teks[0]: "Sebelum",
                    kolom_teks[-1]: "Sesudah"
                }),
                use_container_width=True
            )




# =====================================================
# HASIL DLD (WORD CORRECTION)
# =====================================================
elif menu == "Hasil DLD":

    import os
    import pickle
    import streamlit as st
    import pandas as pd

    st.title("✏️ Word Correction — Damerau Levenshtein Distance (DLD)")
    st.caption("Hasil koreksi kata berbasis DLD per aspek dan per fold")

    BASE_DIR = "visualisasi/dld_hasil"

    # ================= VALIDASI FOLDER =================
    if not os.path.exists(BASE_DIR):
        st.error(f"Folder tidak ditemukan: {BASE_DIR}")
        st.stop()

    # ================= PILIH ASPEK =================
    aspek_list = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    if not aspek_list:
        st.warning("Folder aspek DLD kosong")
        st.stop()

    aspek = st.selectbox("Pilih Aspek", aspek_list)
    aspek_dir = os.path.join(BASE_DIR, aspek)

    # ================= PILIH FOLD =================
    fold_files = sorted([
        f for f in os.listdir(aspek_dir)
        if f.endswith(".pkl") and f.startswith("fold_")
    ])

    if not fold_files:
        st.warning("File fold tidak ditemukan")
        st.stop()

    fold_file = st.selectbox("Pilih Fold", fold_files)

    # ================= LOAD PKL =================
    with open(os.path.join(aspek_dir, fold_file), "rb") as f:
        obj = pickle.load(f)

    data = obj["data"]
    tokencount = data.get("tokencount", {})

    # ================= METRIK TOKEN =================
    st.subheader("📊 Statistik Token")

    col1, col2 = st.columns(2)
    col1.metric(
        "TRAIN Token (Sebelum → Sesudah)",
        f"{tokencount.get('train_before', 0)} → {tokencount.get('train_after', 0)}"
    )
    col2.metric(
        "TEST Token (Sebelum → Sesudah)",
        f"{tokencount.get('test_before', 0)} → {tokencount.get('test_after', 0)}"
    )

    # ================= PILIH DATA =================
    jenis = st.radio("Pilih Data", ["Train", "Test"], horizontal=True)
    df = data["train"] if jenis == "Train" else data["test"]

    if df.empty:
        st.warning("Data kosong")
        st.stop()

    # ================= VALIDASI KOLOM =================
    required_cols = [
        "Ulasan_Stemming",
        "Ulasan_Hasil_DLD",
        "Peta_Koreksi_DLD"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Kolom tidak ditemukan: {missing}")
        st.stop()

    # ================= FORMAT DATA =================
    df_show = df[required_cols].copy()

    df_show["Ulasan_Stemming"] = df_show["Ulasan_Stemming"].apply(lambda x: " ".join(x))
    df_show["Ulasan_Hasil_DLD"] = df_show["Ulasan_Hasil_DLD"].apply(lambda x: " ".join(x))
    df_show["Peta_Koreksi_DLD"] = df_show["Peta_Koreksi_DLD"].astype(str)

    df_show = df_show.rename(columns={
        "Ulasan_Stemming": "Sebelum DLD",
        "Ulasan_Hasil_DLD": "Sesudah DLD",
        "Peta_Koreksi_DLD": "Peta Koreksi"
    })

    st.markdown(
        f"### 📄 {jenis} — Aspek **{aspek}** — {fold_file}"
    )

    st.dataframe(
        df_show,
        use_container_width=True,
        height=520
    )


# =====================================================
# WORDCLOUD TF-IDF
# =====================================================
elif menu == "WordCloud TF-IDF":

    st.title("☁️ WordCloud TF-IDF")

    BASE_DIR = "visualisasi/export_tfidf_pkl"

    # VALIDASI FOLDER
    if not os.path.exists(BASE_DIR):
        st.error("Folder export_tfidf_pkl tidak ditemukan.")
        st.stop()

    # PILIH ASPEK
    aspek_list = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    aspek = st.selectbox("Pilih Aspek", aspek_list)

    # AMBIL MODE N-GRAM (DARI FOLD 1)
    fold1_path = os.path.join(BASE_DIR, aspek, "fold_1.pkl")

    if not os.path.exists(fold1_path):
        st.error("fold_1.pkl tidak ditemukan.")
        st.stop()

    with open(fold1_path, "rb") as f:
        sample_data = pickle.load(f)

    mode_list = sorted(sample_data["modes"].keys())
    mode = st.selectbox("Pilih Jenis N-Gram", mode_list)

    word_scores = defaultdict(float)
    word_counts = defaultdict(int)

    aspek_dir = os.path.join(BASE_DIR, aspek)

    for file in sorted(os.listdir(aspek_dir)):

        if not file.endswith(".pkl"):
            continue

        with open(os.path.join(aspek_dir, file), "rb") as f:
            data = pickle.load(f)

        if mode not in data["modes"]:
            continue

        X = data["modes"][mode]["X_train"]
        vectorizer = data["modes"][mode]["vectorizer"]

        feature_names = vectorizer.get_feature_names_out()
        tfidf_mean_fold = np.asarray(X.mean(axis=0)).flatten()

        for word, score in zip(feature_names, tfidf_mean_fold):
            if score > 0:
                word_scores[word] += score
                word_counts[word] += 1

    if len(word_scores) == 0:
        st.warning("Tidak ada data TF-IDF untuk WordCloud.")
        st.stop()

    # Rata-rata bobot TF-IDF antar fold
    word_freq = {
        word: word_scores[word] / word_counts[word]
        for word in word_scores
    }

# =====================================================
# JUMLAH FITUR N-GRAM (TF-IDF)
# =====================================================
elif menu == "Fitur N-Gram":

    import os
    import pickle
    import pandas as pd
    import streamlit as st

    st.title("📊 Jumlah Fitur N-Gram (TF-IDF)")
    st.caption("Per Aspek, Per Fold, dan Per Mode N-Gram")

    BASE_DIR = "visualisasi/ngram_fitur"

    # ================= PILIH JENIS =================
    jenis = st.radio(
        "Jenis Feature Extraction",
        ["Dengan DLD", "Tanpa DLD"],
        horizontal=True
    )

    if jenis == "Dengan DLD":
        ROOT_DIR = os.path.join(BASE_DIR, "tfidf_dld_hasil")
        GLOBAL_FILE = "tfidf_dld_all.pkl"
    else:
        ROOT_DIR = os.path.join(BASE_DIR, "tfidf_no_dld_hasil")
        GLOBAL_FILE = "tfidf_no_dld_all.pkl"

    # ================= LOAD GLOBAL PKL =================
    global_path = os.path.join(ROOT_DIR, GLOBAL_FILE)

    if not os.path.exists(global_path):
        st.error(f"File global tidak ditemukan: {global_path}")
        st.stop()

    with open(global_path, "rb") as f:
        global_obj = pickle.load(f)

    df_summary = global_obj["summary"].copy()

    # ================= FILTER =================
    aspek_list = sorted(df_summary["Aspek"].unique())
    aspek = st.selectbox("Pilih Aspek", aspek_list)

    fold_option = st.selectbox(
        "Pilih Fold",
        ["Semua Fold"] + sorted(df_summary["Fold"].unique().tolist())
    )

    df_view = df_summary[df_summary["Aspek"] == aspek]

    if fold_option != "Semua Fold":
        df_view = df_view[df_view["Fold"] == fold_option]

    # ================= TABEL =================
    st.markdown("### 📋 Rekap Jumlah Fitur")

    tampil = (
        df_view[[
            "Fold",
            "Mode",
            "Jumlah_Fitur",
            "Train_Shape",
            "Test_Shape"
        ]]
        .sort_values(["Fold", "Mode"])
        .reset_index(drop=True)
    )

    st.dataframe(
        tampil,
        use_container_width=True,
        height=500
    )


# =====================================================
# MATRIKS TF-IDF
# =====================================================
elif menu == "Matriks TF-IDF":

    import os
    import pickle
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.title("🧮 Matriks TF-IDF")
    st.caption("Per Aspek, Fold, dan Mode N-Gram")

    BASE_DIR = "visualisasi/tfidf_matrix"

    # ================= PILIH JENIS =================
    jenis = st.radio(
        "Jenis TF-IDF",
        ["Dengan DLD", "Tanpa DLD"],
        horizontal=True
    )

    if jenis == "Dengan DLD":
        ROOT_DIR = os.path.join(BASE_DIR, "tfidf_dld_matrix")
        GLOBAL_FILE = "tfidf_dld_matrix_all.pkl"
    else:
        ROOT_DIR = os.path.join(BASE_DIR, "tfidf_no_dld_matrix")
        GLOBAL_FILE = "tfidf_no_dld_matrix_all.pkl"

    # ================= VALIDASI =================
    if not os.path.exists(ROOT_DIR):
        st.error(f"Folder tidak ditemukan: {ROOT_DIR}")
        st.stop()

    # ================= PILIH ASPEK =================
    aspek_list = sorted([
        d for d in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, d))
    ])

    aspek = st.selectbox("Pilih Aspek", aspek_list)

    aspek_dir = os.path.join(ROOT_DIR, aspek)

    # ================= PILIH FOLD =================
    fold_list = sorted([
        d for d in os.listdir(aspek_dir)
        if os.path.isdir(os.path.join(aspek_dir, d))
    ])

    fold = st.selectbox("Pilih Fold", fold_list)

    fold_dir = os.path.join(aspek_dir, fold)

    # ================= PILIH MODE =================
    mode_files = sorted([
        f for f in os.listdir(fold_dir)
        if f.endswith(".pkl")
    ])

    mode_file = st.selectbox("Pilih Mode N-Gram", mode_files)

    # ================= LOAD PKL =================
    with open(os.path.join(fold_dir, mode_file), "rb") as f:
        obj = pickle.load(f)

    X_train = obj["X_train"]
    X_test  = obj["X_test"]
    vectorizer = obj["vectorizer"]

    # ================= INFO =================
    st.markdown("### 📌 Informasi Matriks")

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Fitur", X_train.shape[1])
    col2.metric("Train Shape", f"{X_train.shape[0]} × {X_train.shape[1]}")
    col3.metric("Test Shape", f"{X_test.shape[0]} × {X_test.shape[1]}")

    # ================= FEATURE NAMES =================
    with st.expander("🔤 Daftar Fitur (N-Gram)"):
        fitur = vectorizer.get_feature_names_out()
        st.write(pd.DataFrame(fitur, columns=["Fitur"]))

    # ================= PREVIEW MATRIX =================
    st.markdown("### 👀 Contoh Matriks TF-IDF (Train)")

    # Batasi agar tidak berat
    max_row = min(500, X_train.shape[0])
    max_col = min(250, X_train.shape[1])

    preview_matrix = X_train[:max_row, :max_col].toarray()

    df_preview = pd.DataFrame(
        preview_matrix,
        columns=vectorizer.get_feature_names_out()[:max_col]
    )

    st.dataframe(
        df_preview,
        use_container_width=True
    )

    st.caption(
        f"Menampilkan {max_row} baris × {max_col} fitur pertama"
    )


# =====================================================
# UJI N-GRAM Tunggal– PER ASPEK & PER FOLD
# =====================================================
elif menu == "Uji Coba N-Gram Tunggal":

    st.title("📊 Uji Coba N-Gram Tunggal")
    st.caption("Hasil Evaluasi model SVM per aspek, per fold, dan satu jenis N-Gram")

    BASE_DIR = "visualisasi"

    # =========================
    # PILIH MODEL (DLD / NO DLD)
    # =========================
    model_type = st.selectbox(
        "Pilih Model",
        ["SVM + DLD", "SVM Tanpa DLD"]
    )

    model_folder = "svm_dld" if model_type == "SVM + DLD" else "svm_no_dld"

    # =========================
    # PATH NGRAM
    # =========================
    NGRAM_DIR = os.path.join(BASE_DIR, model_folder, "ngram")

    if not os.path.exists(NGRAM_DIR):
        st.error(f"Folder tidak ditemukan: {NGRAM_DIR}")
        st.stop()

    # =========================
    # PILIH ASPEK
    # =========================
    aspek_map = {
        "Aksesibilitas": "aksesibilitas",
        "Amenitas": "amenitas",
        "Daya Tarik": "daya_tarik",
        "Pelayanan Tambahan": "pelayanan_tambahan"
    }

    aspek_label = st.selectbox("Pilih Aspek", list(aspek_map.keys()))
    aspek_folder = aspek_map[aspek_label]

    aspek_path = os.path.join(NGRAM_DIR, aspek_folder)

    if not os.path.exists(aspek_path):
        st.error(f"Folder aspek tidak ditemukan: {aspek_path}")
        st.stop()

    # =========================
    # PILIH FOLD
    # =========================
    folds = sorted(
        [f for f in os.listdir(aspek_path) if f.startswith("fold_")],
        key=lambda x: int(x.split("_")[1])
    )

    if not folds:
        st.warning("Folder fold tidak ditemukan.")
        st.stop()

    fold = st.selectbox("Pilih Fold", folds)

    fold_path = os.path.join(aspek_path, fold)

    # =========================
    # PILIH N-GRAM
    # =========================
    ngram_files = sorted(
        [f for f in os.listdir(fold_path) if f.endswith(".pkl")]
    )

    if not ngram_files:
        st.error("File .pkl tidak ditemukan pada fold ini.")
        st.stop()

    ngram_choice = st.selectbox(
        "Pilih Jenis N-Gram",
        [f.replace(".pkl", "").capitalize() for f in ngram_files]
    )

    pkl_file = ngram_choice.lower() + ".pkl"
    pkl_path = os.path.join(fold_path, pkl_file)

    # =========================
    # LOAD PKL
    # =========================
    data = load_pkl(pkl_path)

    report = data["classification_report"]
    cm = data["confusion_matrix"]

    # =========================
    # METRIK
    # =========================
    st.subheader("📈 Hasil Evaluasi")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{report['accuracy']:.3f}")
    c2.metric("Precision", f"{report['weighted avg']['precision']:.3f}")
    c3.metric("Recall", f"{report['weighted avg']['recall']:.3f}")
    c4.metric("F1-Score", f"{report['weighted avg']['f1-score']:.3f}")

    # =========================
    # CONFUSION MATRIX
    # =========================
    st.subheader("🧩 Confusion Matrix")

    plot_confusion_matrix_small(
        cm,
        ["Negatif", "Positif"],
        f"{aspek_label} | {ngram_choice} | {model_type} | {fold}"
    )




# =====================================================
# UJI Rentang N-GRAM – PER ASPEK & PER FOLD
# =====================================================
elif menu == "Uji Coba Rentang N-Gram":

    st.title("📊 Uji Coba Rentang N-Gram")
    st.caption("Hasil Evaluasi model SVM Rentang N-Gram per aspek dan per fold")

    BASE_DIR = "visualisasi"

    # =========================
    # PILIH MODEL
    # =========================
    model_type = st.selectbox(
        "Pilih Model",
        ["SVM + DLD", "SVM Tanpa DLD"]
    )

    model_folder = "svm_dld" if model_type == "SVM + DLD" else "svm_no_dld"

    # =========================
    # PATH KOMBINASI NGRAM
    # =========================
    COMBO_DIR = os.path.join(BASE_DIR, model_folder, "kombinasi_ngram")

    if not os.path.exists(COMBO_DIR):
        st.error(f"Folder tidak ditemukan: {COMBO_DIR}")
        st.stop()

    # =========================
    # PILIH ASPEK
    # =========================
    aspek_map = {
        "Aksesibilitas": "aksesibilitas",
        "Amenitas": "amenitas",
        "Daya Tarik": "daya_tarik",
        "Pelayanan Tambahan": "pelayanan_tambahan"
    }

    aspek_label = st.selectbox("Pilih Aspek", list(aspek_map.keys()))
    aspek_folder = aspek_map[aspek_label]

    aspek_path = os.path.join(COMBO_DIR, aspek_folder)

    if not os.path.exists(aspek_path):
        st.error(f"Folder aspek tidak ditemukan: {aspek_path}")
        st.stop()

    # =========================
    # PILIH FOLD
    # =========================
    folds = sorted(
        [f for f in os.listdir(aspek_path) if f.startswith("fold_")],
        key=lambda x: int(x.split("_")[1])
    )

    if not folds:
        st.warning("Folder fold tidak ditemukan.")
        st.stop()

    fold = st.selectbox("Pilih Fold", folds)
    fold_path = os.path.join(aspek_path, fold)

    # =========================
    # PILIH KOMBINASI N-GRAM
    # =========================
    combo_files = sorted(
        [f for f in os.listdir(fold_path) if f.endswith(".pkl")]
    )

    if not combo_files:
        st.error("File kombinasi N-Gram tidak ditemukan.")
        st.stop()

    combo_choice = st.selectbox(
        "Pilih Kombinasi N-Gram",
        [f.replace(".pkl", "").replace("_", " + ").title() for f in combo_files]
    )

    # Kembalikan ke nama file
    pkl_file = combo_choice.lower().replace(" + ", "_") + ".pkl"
    pkl_path = os.path.join(fold_path, pkl_file)

    # =========================
    # LOAD PKL
    # =========================
    data = load_pkl(pkl_path)

    report = data["classification_report"]
    cm = data["confusion_matrix"]

    # =========================
    # METRIK
    # =========================
    st.subheader("📈 Hasil Evaluasi")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{report['accuracy']:.3f}")
    c2.metric("Precision", f"{report['weighted avg']['precision']:.3f}")
    c3.metric("Recall", f"{report['weighted avg']['recall']:.3f}")
    c4.metric("F1-Score", f"{report['weighted avg']['f1-score']:.3f}")

    # =========================
    # CONFUSION MATRIX
    # =========================
    st.subheader("🧩 Confusion Matrix")

    plot_confusion_matrix_small(
        cm,
        ["Negatif", "Positif"],
        f"{aspek_label} | {combo_choice} | {model_type} | {fold}"
    )




# =====================================================
# PERBANDINGAN PENGGUNAAN DLD
# =====================================================
elif menu == "Perbandingan Penggunaan DLD":

    st.title("📊 Perbandingan Penggunaan DLD")
    st.caption(
        "Perbandingan performa terbaik "
        "antara model SVM dengan DLD dan tanpa DLD "
        "berdasarkan kombinasi N-Gram dan aspek"
    )

    # ===============================
    # LOAD DATA PKL
    # ===============================
    with open("visualisasi/ringkasan_best_fold_dld_vs_no_dld.pkl", "rb") as f:
        df_summary = pickle.load(f)

    with open("visualisasi/data_grafik_dld_vs_no_dld.pkl", "rb") as f:
        graph_data = pickle.load(f)

    modes_selected = graph_data["modes"]

    # ===============================
    # FILTER ASPEK
    # ===============================
    aspek_list = sorted(df_summary["Aspek"].unique())
    aspek_filter = st.multiselect(
        "Filter Aspek",
        aspek_list,
        default=aspek_list
    )

    df_show = df_summary[df_summary["Aspek"].isin(aspek_filter)]

    # ===============================
    # TABEL RINGKASAN
    # ===============================
    st.subheader("📋 Tabel Ringkasan Performa Terbaik")

    df_table = df_show.drop(columns=["Fold"], errors="ignore")

    styled_df = (
        df_table
        .sort_values(["Aspek", "Mode", "DLD", "Accuracy"])
        .style
        .set_properties(**{
            "text-align": "center",
            "vertical-align": "middle"
        })
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]}
        ])
    )

    st.dataframe(styled_df, use_container_width=True)

    # ===============================
    # GRAFIK PERBANDINGAN
    # ===============================
    st.subheader("📈 Grafik Perbandingan Akurasi Terbaik")

    aspek_order = df_show["Aspek"].unique()
    x = np.arange(len(aspek_order))
    width = 0.10

    fig, ax = plt.subplots(figsize=(15, 6))

    for i, mode in enumerate(modes_selected):

        # ---------- DLD ----------
        df_dld = (
            df_show[
                (df_show["Mode"] == mode) &
                (df_show["DLD"] == "Ya")
            ]
            .set_index("Aspek")
            .reindex(aspek_order)
        )

        bars_dld = ax.bar(
            x + i * 2 * width,
            df_dld["Accuracy"],
            width,
            label=f"{mode} (DLD)"
        )

        for idx, bar in enumerate(bars_dld):
            if not pd.isna(df_dld["Accuracy"].iloc[idx]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,   # 🔧 JARAK AMAN
                    f"{bar.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    -0.08,
                    f"F{int(df_dld['Fold'].iloc[idx])}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    transform=ax.get_xaxis_transform()
                )

        # ---------- TANPA DLD ----------
        df_no = (
            df_show[
                (df_show["Mode"] == mode) &
                (df_show["DLD"] == "Tidak")
            ]
            .set_index("Aspek")
            .reindex(aspek_order)
        )

        bars_no = ax.bar(
            x + i * 2 * width + width,
            df_no["Accuracy"],
            width,
            label=f"{mode} (Tanpa DLD)",
            hatch="///",
            fill=False
        )

        for idx, bar in enumerate(bars_no):
            if not pd.isna(df_no["Accuracy"].iloc[idx]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,   # 🔧 JARAK AMAN
                    f"{bar.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    -0.13,
                    f"F{int(df_no['Fold'].iloc[idx])}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    transform=ax.get_xaxis_transform()
                )

    # ===============================
    # BENAHI TAMPILAN GRAFIK (PENTING)
    # ===============================
    ax.set_ylim(0, 1.05)              # 🔥 biar angka tidak nabrak
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    ax.set_xlabel("Aspek")
    ax.set_ylabel("Accuracy Terbaik")
    ax.set_title("Perbandingan Akurasi Terbaik\nDLD vs Tanpa DLD")

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(aspek_order)

    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18)
    )

    st.pyplot(fig)
