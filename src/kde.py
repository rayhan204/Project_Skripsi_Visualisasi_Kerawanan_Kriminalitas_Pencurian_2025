import numpy as np
import pandas as pd
import geopandas as gpd
import os
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

# =========================
# PARAMETER KDE
# =========================
BANDWIDTH = 1500
GRID_RES = 100
CRS_UTM = "EPSG:32749"


# =========================
# GAUSSIAN KERNEL
# =========================
def gaussian_kernel(u):
    return (1 / (2 * np.pi)) * np.exp(-0.5 * u**2)


# =========================
# KDE LEAVE ONE OUT
# =========================
def kde_leave_one_out(points, h):

    n = len(points)
    densities = []

    for i in range(n):

        x_i, y_i = points[i]
        others = np.delete(points, i, axis=0)

        dist = np.sqrt(
            (others[:, 0] - x_i) ** 2 +
            (others[:, 1] - y_i) ** 2
        )

        u = dist / h

        density = np.sum(gaussian_kernel(u)) / ((n - 1) * h**2)

        densities.append(density)

    return np.array(densities)


# =========================
# KDE GRID SURFACE
# =========================
def generate_kde_grid(points, desa_geo):

    desa_utm = desa_geo.to_crs(CRS_UTM)

    minx, miny, maxx, maxy = desa_utm.total_bounds

    x_grid = np.arange(minx, maxx, GRID_RES)
    y_grid = np.arange(miny, maxy, GRID_RES)

    grid_x, grid_y = np.meshgrid(x_grid, y_grid)

    density_grid = np.zeros(grid_x.shape)

    n = len(points)

    for x_i, y_i in points:

        dist = np.sqrt((grid_x - x_i)**2 + (grid_y - y_i)**2)

        u = dist / BANDWIDTH

        density_grid += gaussian_kernel(u)

    density_grid = density_grid / (n * BANDWIDTH**2)

    return density_grid, minx, miny, maxx, maxy, desa_utm


# =========================
# EXPORT RASTER
# =========================
def export_kde_raster(density_grid, minx, maxy):

    transform = from_origin(minx, maxy, GRID_RES, GRID_RES)

    with rasterio.open(
        "output/kde_surface.tif",
        "w",
        driver="GTiff",
        height=density_grid.shape[0],
        width=density_grid.shape[1],
        count=1,
        dtype="float32",
        crs=CRS_UTM,
        transform=transform
    ) as dst:

        dst.write(density_grid.astype("float32"), 1)


# =========================
# EXPORT PNG
# =========================
def export_png_map(density_grid, desa_utm, minx, miny, maxx, maxy):

    kabupaten = desa_utm.dissolve()

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        density_grid,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
        cmap="Reds",
        alpha=0.9
    )

    kabupaten.boundary.plot(ax=ax, edgecolor="black", linewidth=1)

    ax.set_title("Kernel Density Estimation")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Kepadatan KDE")

    plt.tight_layout()
    plt.savefig("output/kde_surface_admin.png", dpi=300)
    plt.close()


# =========================
# MAIN KDE PROCESS
# =========================
def run_kde():

    os.makedirs("output", exist_ok=True)

    crime = pd.read_csv("data/processed/kriminalitas_utm.csv")

    points = crime[["x", "y"]].values

    # KDE per titik
    crime["kepadatan_KDE"] = kde_leave_one_out(points, BANDWIDTH)

    crime.rename(
        columns={"x": "X_UTM", "y": "Y_UTM"}
    )[[
        "longitude",
        "latitude",
        "desa",
        "X_UTM",
        "Y_UTM",
        "kepadatan_KDE"
    ]].to_csv(
        "output/hasil_kde_per_titik.csv",
        index=False
    )

    # Load batas desa
    desa_geo = gpd.read_file("data/raw/batas_desa_purbalingga.geojson")

    density_grid, minx, miny, maxx, maxy, desa_utm = generate_kde_grid(points, desa_geo)

    export_kde_raster(density_grid, minx, maxy)

    export_png_map(density_grid, desa_utm, minx, miny, maxx, maxy)

    print("\n======================================")
    print("KDE PROCESS SELESAI")
    print("======================================")
    print(" output/hasil_kde_per_titik.csv")
    print(" output/kde_surface.tif")
    print(" output/kde_surface_admin.png")
    print("======================================\n")

# =========================
# EKSEKUSI PROGRAM
# =========================
if __name__ == "__main__":
    run_kde()