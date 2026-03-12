# Visualisasi Kerawanan Kriminalitas Pencurian Menggunakan KDE  
## Kabupaten Purbalingga – Tahun 2025

Project ini merupakan implementasi analisis spasial untuk memetakan **kerawanan kriminalitas pencurian** di Kabupaten Purbalingga menggunakan metode **Kernel Density Estimation (KDE)** berbasis Python, dengan luaran berupa **visualisasi peta berbasis web**.

---

## 📌 Tujuan Project
- Memvisualisasikan sebaran titik kejadian kriminalitas pencurian
- Mengidentifikasi pola kepadatan (hotspot) menggunakan KDE
- Mengklasifikasikan tingkat kerawanan menjadi 5 kategori
- Menyajikan hasil analisis dalam bentuk peta interaktif berbasis web

---


---

## 📂 Penjelasan File dan Fitur

### 🔹 `data/raw/`
Berisi **data mentah**:
- `kriminalitas_pencurian_purbalingga_2025.csv`  
  → Data titik kejadian kriminalitas (longitude, latitude, desa, jumlah)
- `batas_desa_purbalingga.geojson`  
  → Batas administrasi desa Kabupaten Purbalingga

---

### 🔹 `data/processed/`
Berisi **data hasil pra-pemrosesan**:
- `kriminalitas_utm.csv`  
  → Data kriminalitas yang sudah diproyeksikan ke UTM (meter)
- `desa_centroid.geojson`  
  → Titik centroid setiap desa (digunakan untuk agregasi & evaluasi)

---

### 🔹 `src/preprocess.py`
**Fitur:**
- Membaca data CSV & GeoJSON
- Transformasi CRS dari EPSG:4326 ke EPSG:32749 (UTM)
- Menyiapkan data siap analisis KDE

---

### 🔹 `src/kde.py`
**Fitur utama analisis:**
- Menghitung Kernel Density Estimation (KDE)
- Mengagregasikan nilai kepadatan ke wilayah desa
- Mengklasifikasikan tingkat kerawanan
- Menghasilkan output:
  - `output/kerawanan_desa.geojson`

---

### 🔹 `src/classification.py`
**Fitur:**
- Klasifikasi 5 tingkat kerawanan:
  - Sangat Rawan
  - Rawan
  - Cukup Rawan
  - Cukup Aman
  - Aman

---

### 🔹 `src/evaluation.py`
**Fitur:**
- Evaluasi model KDE menggunakan:
  - Root Mean Square Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)

---

### 🔹 `web/app.py`
**Fitur visualisasi web:**
- Peta interaktif berbasis Streamlit & Folium
- Menampilkan:
  - Titik kejadian kriminalitas
  - Heatmap KDE (efek sebaran cahaya/bandwidth)
  - Batas desa
- Bandwidth visual dapat diatur melalui sidebar

---

## Cara Menjalankan Project (URUTAN WAJIB)
1️⃣ Pra-pemrosesan Data
python src/preprocess.py

Output:
data/processed/kriminalitas_utm.csv
data/processed/desa_centroid.geojson

2️⃣ Analisis KDE & Klasifikasi Kerawanan
python src/kde.py
Output:
output/kerawanan_desa.geojson

3️⃣ Evaluasi Model
python src/evaluation.py

4️⃣ Menjalankan Visualisasi Web
streamlit run web/app.py


## urutan menjalankan project
python src/preprocess.py
python src/kde.py
python src/classification.py
python src/evaluation_pai.py
streamlit run web/app.py
