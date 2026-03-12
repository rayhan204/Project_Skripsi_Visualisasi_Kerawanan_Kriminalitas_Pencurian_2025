import geopandas as gpd
import pandas as pd

def preprocess():
    # =========================
    # DATA KRIMINALITAS
    # =========================
    crime = pd.read_csv(
        "data/raw/kriminalitas_pencurian_purbalingga_2025.csv"
    )

    gdf_crime = gpd.GeoDataFrame(
        crime,
        geometry=gpd.points_from_xy(
            crime["longitude"],
            crime["latitude"]
        ),
        crs="EPSG:4326"
    )

    gdf_crime = gdf_crime.to_crs(epsg=32749)
    gdf_crime["x"] = gdf_crime.geometry.x
    gdf_crime["y"] = gdf_crime.geometry.y

    gdf_crime.to_csv(
        "data/processed/kriminalitas_utm.csv",
        index=False
    )

    # =========================
    # DATA DESA
    # =========================
    desa = gpd.read_file(
        "data/raw/batas_desa_purbalingga.geojson"
    ).to_crs(epsg=32749)

    desa_centroid = gpd.GeoDataFrame(
        desa.drop(columns="geometry"),
        geometry=desa.geometry.centroid,
        crs=desa.crs
    )

    desa_centroid.to_file(
        "data/processed/desa_centroid.geojson",
        driver="GeoJSON"
    )

    print("Preprocess selesai")

if __name__ == "__main__":
    preprocess()
