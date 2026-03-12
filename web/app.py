import streamlit as st
import geopandas as gpd
import folium
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import branca.colormap as cm
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
# FUNGSI KLASIFIKASI
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


# =====================================================
# LOAD DATA
# =====================================================
crime = pd.read_csv("output/hasil_kde_per_titik.csv")
crime = crime.dropna(subset=["latitude","longitude"])

desa = gpd.read_file("data/raw/batas_desa_purbalingga.geojson")

# =====================================================
# AGREGASI DESA
# =====================================================
desa_kde = (
    crime.groupby("desa",as_index=False)
    .agg(
        rata_kde=("kepadatan_KDE","mean"),
        jumlah_kasus=("kepadatan_KDE","count")
    )
)

desa_kde["kelas_kerawanan"] = classify(desa_kde["rata_kde"].values)

# =====================================================
# MERGE KE TITIK
# =====================================================
crime = crime.merge(
    desa_kde[["desa","kelas_kerawanan","jumlah_kasus"]],
    on="desa",
    how="left"
)

# =====================================================
# TOOLTIP TITIK
# =====================================================
def tooltip_point(row):

    return f"""
    <b>Desa:</b> {row['desa']}<br>
    <b>Tingkat Kerawanan:</b> {row['kelas_kerawanan']}<br>
    <b>Nilai KDE:</b> {row['kepadatan_KDE']:.2e}<br>
    <b>Jumlah Kasus Desa:</b> {int(row['jumlah_kasus'])}
    """


# =====================================================
# MERGE DATA DESA
# =====================================================
desa["desa"] = desa["DESA"].str.lower().str.strip()
desa_kde["desa"] = desa_kde["desa"].str.lower().str.strip()

desa = desa.merge(desa_kde,on="desa",how="left")

desa["rata_kde"] = desa["rata_kde"].fillna(0)
desa["jumlah_kasus"] = desa["jumlah_kasus"].fillna(0)
desa["kelas_kerawanan"] = desa["kelas_kerawanan"].fillna("Aman")


# =====================================================
# LOAD & REPROJECT RASTER
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

gamma=0.45
kde_enhanced = np.power(kde_norm,gamma)

left = transform.c
top = transform.f
right = left + transform.a * width
bottom = top + transform.e * height

bounds_latlon=[[bottom,left],[top,right]]

# =====================================================
# SIMPAN PNG HEATMAP
# =====================================================
png_path="output/kde_surface_visual.png"

fig,ax=plt.subplots(figsize=(6,6))

ax.imshow(kde_enhanced,cmap="hot",origin="lower",alpha=kde_enhanced)
ax.axis("off")

plt.tight_layout()
plt.savefig(png_path,dpi=300,transparent=True)
plt.close(fig)

# =====================================================
# MAP
# =====================================================
m=folium.Map(location=[-7.39,109.36],zoom_start=11,tiles=None)

# =====================================================
# BASEMAP (TIDAK DIUBAH)
# =====================================================
folium.TileLayer("CartoDB positron",name="Carto Light").add_to(m)

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
# LAYER HEATMAP
# =====================================================
heat_layer = folium.FeatureGroup(name="Permukaan kerawanan (indeks 0–1)")

ImageOverlay(
    image=png_path,
    bounds=bounds_latlon,
    opacity=1
).add_to(heat_layer)

heat_layer.add_to(m)

# =====================================================
# LAYER BATAS DESA
# =====================================================
desa_wgs = desa.to_crs("EPSG:4326")

desa_layer = folium.FeatureGroup(name="Batas Desa (hover)")

folium.GeoJson(
    desa_wgs,
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
# LAYER TITIK
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

titik_layer.add_to(m)

# =====================================================
# LEGEND
# =====================================================
colormap = cm.LinearColormap(
    colors=["#ffffcc","#ffeda0","#feb24c","#f03b20","#bd0026"],
    vmin=float(np.min(kde_wgs84)),
    vmax=float(np.max(kde_wgs84)),
    caption="Kepadatan KDE"
)

colormap.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, use_container_width=True, height=550)