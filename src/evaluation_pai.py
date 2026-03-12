import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

# =====================================
# LOAD RASTER KDE
# =====================================
raster_path = "output/kde_surface.tif"

with rasterio.open(raster_path) as src:
    kde = src.read(1)
    transform = src.transform
    crs = src.crs
    pixel_size_x = src.res[0]
    pixel_size_y = src.res[1]

pixel_area = abs(pixel_size_x * pixel_size_y)

# =====================================
# LOAD TITIK KRIMINALITAS
# =====================================
points_df = pd.read_csv("output/hasil_kde_per_titik.csv")

gdf_points = gpd.GeoDataFrame(
    points_df,
    geometry=gpd.points_from_xy(points_df["X_UTM"], points_df["Y_UTM"]),
    crs=crs
)

N = len(gdf_points)

print("===================================")
print("Evaluasi PAI Otomatis 50% - 85%")
print("Total Kejahatan (N):", N)
print("Luas Total Wilayah (A): akan dihitung di loop")
print("===================================")

# =====================================
# LOOP THRESHOLD
# =====================================
results = []

for threshold in range(50, 90, 5):

    threshold_value = np.percentile(kde[~np.isnan(kde)], threshold)
    hotspot_mask = kde >= threshold_value

    A = np.sum(~np.isnan(kde)) * pixel_area
    a = np.sum(hotspot_mask) * pixel_area

    def point_in_hotspot(point):
        row, col = rasterio.transform.rowcol(transform, point.x, point.y)
        try:
            return hotspot_mask[row, col]
        except:
            return False

    gdf_points["in_hotspot"] = gdf_points["geometry"].apply(point_in_hotspot)
    n = gdf_points["in_hotspot"].sum()

    accuracy = n / N
    area_percentage = a / A
    pai = accuracy / area_percentage if area_percentage != 0 else 0

    # PRINT DETAIL ANGKA
    print("--------------------------------------------------")
    print(f"Threshold: {threshold}%")
    print(f"N (Total Titik): {N}")
    print(f"n (Titik dalam Hotspot): {n}")
    print(f"A (Luas Total Wilayah m2): {A:,.0f}")
    print(f"a (Luas Hotspot m2): {a:,.0f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Area %: {area_percentage:.4f}")
    print(f"PAI: {pai:.4f}")
    print("--------------------------------------------------")

    results.append({
        "Threshold": threshold,
        "N": N,
        "n": int(n),
        "A_m2": A,
        "a_m2": a,
        "Accuracy": round(accuracy, 4),
        "Area_Percentage": round(area_percentage, 4),
        "PAI": round(pai, 4)
    })

# =====================================
# TABEL HASIL
# =====================================
df_results = pd.DataFrame(results)

print("\nTABEL HASIL EVALUASI:")
print(df_results)

df_results.to_csv("output/hasil_evaluasi_pai_lengkap.csv", index=False)

# =====================================
# PLOT KURVA PAI
# =====================================
plt.figure(figsize=(8, 6))
plt.plot(df_results["Area_Percentage"], df_results["PAI"], marker='o')
plt.xlabel("Area Percentage")
plt.ylabel("PAI")
plt.title("Kurva PAI vs Area Percentage")
plt.grid(True)

plt.tight_layout()
plt.savefig("output/kurva_pai.png", dpi=300)
plt.show()

print("===================================")
print("File disimpan:")
print("- output/hasil_evaluasi_pai_lengkap.csv")
print("- output/kurva_pai.png")
print("===================================")