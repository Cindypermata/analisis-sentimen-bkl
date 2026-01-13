# KEYWORDS ASPEK
aspek_keywords = {
    "Daya Tarik": [
        "pantai", "laut", "ombak", "pasir", "pemandangan", "sunset", "sunrise", "foto", "spot", "indah", "bagus", "bersih", "keren", "cantik", "luas", "asri", "ramai", "alami"
    ],
    "Aksesibilitas": [
        "jalan", "akses", "rute", "lokasi", "arah", "dekat", "jauh", "mudah", "sulit", "macet", "berlubang", "aspal", "motor", "mobil", "maps"
    ],
    "Amenitas": [
        "toilet", "wc", "warung", "gazebo", "makan", "minum", "penginapan", "musholla", "sampah", "meja", "kursi", "parkir", "kafe"
    ],
    "Pelayanan Tambahan": [
        "petugas", "penjaga", "pengelola", "sewa", "fotografer", "tiket", "satpam", "keamanan", "kasir"
    ]
}


# DETEKSI ASPEK
def detect_aspek(text):
    text = text.lower()
    aspek_ditemukan = []

    for aspek, keywords in aspek_keywords.items():
        for k in keywords:
            if k in text:
                aspek_ditemukan.append(aspek)
                break

    if len(aspek_ditemukan) == 0:
        return "Tidak Teridentifikasi"

    return ", ".join(aspek_ditemukan)
