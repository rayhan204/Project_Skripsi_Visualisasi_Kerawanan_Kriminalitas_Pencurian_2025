import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# =========================
# KLASIFIKASI KUANTIL
# =========================
def classify(values):

    q = np.quantile(values, [0.2, 0.4, 0.6, 0.8])

    labels = []

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


# =========================
# AGREGASI DESA
# =========================
def agregasi_desa(df):

    desa_kde = (
        df.groupby("desa", as_index=False)
        .agg(
            rata_kde=("kepadatan_KDE", "mean"),
            jumlah_kasus=("kepadatan_KDE", "count")
        )
    )

    desa_kde["kelas_kerawanan"] = classify(
        desa_kde["rata_kde"].values
    )

    desa_kde = desa_kde.sort_values(
        "rata_kde",
        ascending=False
    )

    return desa_kde


# =========================
# GRAFIK TOP 10
# =========================
def plot_top10(desa_kde):

    top10 = desa_kde.head(10).iloc[::-1]

    plt.figure(figsize=(10, 6))

    bars = plt.barh(
        top10["desa"],
        top10["rata_kde"],
        color="#d7191c"
    )

    for bar, kasus, kde in zip(
        bars,
        top10["jumlah_kasus"],
        top10["rata_kde"]
    ):

        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height()/2,
            f" {kde:.2E} | {kasus} kasus",
            va="center",
            fontsize=9
        )

    plt.xlabel("Nilai KDE")
    plt.title("Top 10 Desa Paling Rawan")

    plt.tight_layout()

    plt.savefig(
        "output/top10_desa_kerawanan_kde.png",
        dpi=300
    )

    plt.close()


# =========================
# MAIN CLASSIFICATION
# =========================
def run_classification():

    os.makedirs("output", exist_ok=True)

    df = pd.read_csv(
        "output/hasil_kde_per_titik.csv"
    )

    desa_kde = agregasi_desa(df)

    desa_kde.to_csv(
        "output/rekap_kerawanan_desa_kde.csv",
        index=False
    )

    plot_top10(desa_kde)

    print("\n======================================")
    print("KLASIFIKASI KERAWANAN SELESAI")
    print("======================================")
    print("   output/rekap_kerawanan_desa_kde.csv")
    print("   output/top10_desa_kerawanan_kde.png")
    print("======================================\n")


# =========================
# EKSEKUSI
# =========================
if __name__ == "__main__":
    run_classification()
