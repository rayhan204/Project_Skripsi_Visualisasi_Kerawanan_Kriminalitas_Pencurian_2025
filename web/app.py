import streamlit as st
import geopandas as gpd
import folium
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.mask

# =====================================================
# KONFIGURASI
# =====================================================
st.set_page_config(layout="wide")

st.markdown("""
<h2>Webmap Detail Hotspot Kriminalitas Pencurian Di Purbalingga 2025</h2>
<p>Kernel Density Estimation (KDE)</p>
""", unsafe_allow_html=True)

# =====================================================
# FUNGSI
# =====================================================
def classify(values):
    q = np.quantile(values,[0.2,0.4,0.6,0.8])
    labels=[]
    for v in values:
        if v >= q[3]:
            labels.append("Sangat Rawan")
        elif v >= q[2]:
            labels.append("Rawan")
        elif v >= q[1]:
            labels.append("Cukup Rawan")
        elif v >= q[0]:
            labels.append("Cukup Aman")
        else:
            labels.append("Aman")
    return labels

# kode desa (manual biar tidak bentrok)
kode_desa = {
    "kalialang": "KA",
    "kalikabong": "KK",
    "purbalingga lor": "PL",
    "bantarbarang": "BB",
    "brobot": "BR",
    "kradenan": "KR",
    "bojong": "BJG",
    "kembaran kulon": "KKL",
    "karangjambe": "KJ",
    "bojongsari": "BJS",
    "sokanegara": "SN",
    "tidu": "TD",
    "bukateja": "BT",
    "pekiringan": "PK",
    "penaruban": "PN",
    "panunggalan": "PG",
    "rabak": "RB",
    "karanganyar": "KA2",
    "sinduraja": "SR",
    "sangkanayu": "SK"
}

def get_kode(nama):
    return kode_desa.get(nama.lower(), nama[:2].upper())

# =====================================================
# LOAD DATA
# =====================================================
crime = pd.read_csv("output/hasil_kde_per_titik.csv")
crime = crime.dropna(subset=["latitude","longitude"])

desa = gpd.read_file("data/raw/batas_desa_purbalingga.geojson")

# =====================================================
# AGREGASI
# =====================================================
desa_kde = (
    crime.groupby("desa",as_index=False)
    .agg(
        rata_kde=("kepadatan_KDE","mean"),
        jumlah_kasus=("kepadatan_KDE","count")
    )
)

desa_kde["kelas_kerawanan"] = classify(desa_kde["rata_kde"].values)

crime = crime.merge(
    desa_kde[["desa","kelas_kerawanan","jumlah_kasus"]],
    on="desa",
    how="left"
)

# =====================================================
# TOOLTIP
# =====================================================
def tooltip_point(row):
    return f"""
    <b>Desa:</b> {row['desa']}<br>
    <b>Kode:</b> {get_kode(row['desa'])}<br>
    <b>Tingkat Kerawanan:</b> {row['kelas_kerawanan']}<br>
    <b>Nilai KDE:</b> {row['kepadatan_KDE']:.2e}<br>
    <b>Jumlah Kasus Desa:</b> {int(row['jumlah_kasus'])}
    """

# =====================================================
# GEODATA
# =====================================================
desa["desa"] = desa["DESA"].str.lower().str.strip()
desa_kde["desa"] = desa_kde["desa"].str.lower().str.strip()

desa = desa.merge(desa_kde,on="desa",how="left")
desa["rata_kde"] = desa["rata_kde"].fillna(0)
desa["jumlah_kasus"] = desa["jumlah_kasus"].fillna(0)
desa["kelas_kerawanan"] = desa["kelas_kerawanan"].fillna("Aman")

# =====================================================
# RASTER KDE
# =====================================================
tif_path="output/kde_surface.tif"

with rasterio.open(tif_path) as src:
    desa_utm = desa.to_crs(src.crs)
    kabupaten = desa_utm.dissolve()

    masked, masked_transform = rasterio.mask.mask(
        src, kabupaten.geometry, crop=True
    )

    kde = masked[0]
    src_crs = src.crs

dst_crs="EPSG:4326"

transform,width,height = calculate_default_transform(
    src_crs,dst_crs,
    kde.shape[1],kde.shape[0],
    *kabupaten.total_bounds
)

kde_wgs84=np.empty((height,width),dtype=np.float32)

reproject(
    source=kde,
    destination=kde_wgs84,
    src_transform=masked_transform,
    src_crs=src_crs,
    dst_transform=transform,
    dst_crs=dst_crs,
    resampling=Resampling.bilinear
)

kde_wgs84=np.nan_to_num(kde_wgs84)
kde_norm = kde_wgs84/np.max(kde_wgs84)
kde_enhanced = np.power(kde_norm,0.45)

left = transform.c
top = transform.f
right = left + transform.a * width
bottom = top + transform.e * height

bounds_latlon=[[bottom,left],[top,right]]

# =====================================================
# SIMPAN PNG
# =====================================================
png_path="output/kde_surface_visual.png"

fig,ax=plt.subplots(figsize=(6,6))
ax.imshow(kde_enhanced,cmap="hot",origin="lower",alpha=kde_enhanced)
ax.axis("off")
plt.savefig(png_path,dpi=300,transparent=True)
plt.close(fig)

# =====================================================
# MAP
# =====================================================
m=folium.Map(location=[-7.39,109.36],zoom_start=11,tiles=None)

# =====================================================
# BASEMAP (TIDAK DIUBAH)
# =====================================================
folium.TileLayer("CartoDB positron", name="Carto Light").add_to(m)

folium.TileLayer(
    tiles="https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google",
    name="Google Satellite",
    subdomains=["mt0","mt1","mt2","mt3"],
).add_to(m)

folium.TileLayer(
    tiles="https://{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google",
    name="Google Hybrid",
    subdomains=["mt0","mt1","mt2","mt3"],
).add_to(m)

# =====================================================
# HEATMAP (LAYER)
# =====================================================
heat_layer = folium.FeatureGroup(name="Permukaan kerawanan")

ImageOverlay(
    image=png_path,
    bounds=bounds_latlon,
    opacity=1
).add_to(heat_layer)

heat_layer.add_to(m)

# =====================================================
# BATAS DESA (LAYER)
# =====================================================
desa_layer = folium.FeatureGroup(name="Batas Desa (hover)")

folium.GeoJson(
    desa.to_crs("EPSG:4326"),
    style_function=lambda x:{
        "fillColor":"none",
        "color":"#00b3ff",
        "weight":1
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["DESA","kelas_kerawanan","jumlah_kasus","rata_kde"],
        aliases=["Desa","Kerawanan","Jumlah Kasus","Rata KDE"],
        localize=True
    )
).add_to(desa_layer)

desa_layer.add_to(m)

# =====================================================
# TITIK + LABEL (LAYER)
# =====================================================
titik_layer = folium.FeatureGroup(name="Titik & Nilai (hover)")

for _,row in crime.iterrows():

    folium.CircleMarker(
        location=[row["latitude"],row["longitude"]],
        radius=4,
        color="black",
        fill=True,
        fill_color="black",
        fill_opacity=0.9,
        tooltip=tooltip_point(row)
    ).add_to(titik_layer)

    folium.Marker(
        location=[row["latitude"],row["longitude"]],
        icon=folium.DivIcon(
            html=f"""
            <div style="
                font-size:9px;
                font-weight:bold;
                background-color:white;
                padding:1px 3px;
                border-radius:3px;
                transform: translate(-50%, -120%);
                pointer-events: none;
            ">
                {get_kode(row['desa'])}
            </div>
            """
        )
    ).add_to(titik_layer)

titik_layer.add_to(m)

# =====================================================
# LEGEND BARU
# =====================================================
legend_items = ""

for _, row in desa_kde.iterrows():
    nama = row["desa"]
    kode = get_kode(nama)
    kelas = row["kelas_kerawanan"]

    legend_items += f"{kode} ({nama.title()}) - {kelas}<br>"

legend_html = f"""
<div style="
position: fixed;
top: 80px;
left: 30px;
width: 250px;
background-color: white;
border:2px solid grey;
z-index:9999;
font-size:13px;
padding: 10px;
max-height:300px;
overflow-y:auto;
box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">

<b>Label Desa & Kerawanan</b><br><br>

{legend_items}

</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

# =====================================================
# CONTROL
# =====================================================
folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, use_container_width=True, height=550)