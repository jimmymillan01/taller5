"""
Taller 5 - Análisis de tráfico LAN y predicción bajo congestión
Procesa las salidas de texto de iperf3 (traza5/10/15.txt), construye ventanas.csv,
hace el análisis exploratorio, ajusta la regresión lineal y registra la predicción
para un escenario de carga alta.

Uso:  python analisis_taller5.py            (desde la carpeta del taller)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).parent
DATOS = BASE / "datos"
FIG = BASE / "figuras"
FIG.mkdir(exist_ok=True)

# Orden real de ejecución inferido del puerto origen (57406 < 57418 < 57435)
TRAZAS = {5: "traza5.txt", 10: "traza10.txt", 15: "traza15.txt"}
VENTANA_S = 60
MSS = 1448  # bytes de payload por segmento TCP típico en Ethernet/Wi-Fi

UNIDADES = {"Bytes": 1, "KBytes": 1024, "MBytes": 1024**2, "GBytes": 1024**3}
UNID_BITS = {"bits/sec": 1e-6, "Kbits/sec": 1e-3, "Mbits/sec": 1, "Gbits/sec": 1e3}
PAT = re.compile(
    r"\[\s*\d+\]\s+([\d.]+)-([\d.]+)\s+sec\s+([\d.]+)\s+(\w+)\s+([\d.]+)\s+([\w/]+)"
    r"\s+(\d+)\s+([\d.]+)\s+(\w+)"
)


def leer_iperf(ruta: Path, objetivo: int) -> pd.DataFrame:
    """Devuelve una fila por intervalo de 1 s (se excluye el resumen final)."""
    texto = ruta.read_text()
    puerto = int(re.search(r"local \S+ port (\d+)", texto).group(1))
    filas = []
    for m in PAT.finditer(texto):
        ini, fin, tr, tru, br, bru, retr, cw, cwu = m.groups()
        filas.append({
            "escenario_mbps": objetivo,
            "puerto_origen": puerto,
            "t_ini": float(ini), "t_fin": float(fin),
            "bytes": float(tr) * UNIDADES[tru],
            "mbps": float(br) * UNID_BITS[bru],
            "retr": int(retr),
            "cwnd_kb": float(cw) * UNIDADES[cwu] / 1024,
        })
    # resumen del emisor (control de calidad)
    res = re.search(r"0\.00-([\d.]+)\s+sec\s+([\d.]+)\s+MBytes\s+([\d.]+)\s+Mbits/sec\s+(\d+)\s+sender", texto)
    df = pd.DataFrame(filas)
    df.attrs["resumen"] = {"dur": float(res.group(1)), "MB": float(res.group(2)),
                           "mbps": float(res.group(3)), "retr": int(res.group(4))}
    return df


# ---------------------------------------------------------------- 1. lectura
seg = []
for obj, nombre in TRAZAS.items():
    d = leer_iperf(DATOS / nombre, obj)
    r = d.attrs["resumen"]
    # control de calidad: la suma por segundo debe coincidir con el resumen de iperf3
    print(f"[QC] {nombre}: {len(d)} intervalos | suma={d.bytes.sum()/2**20:.1f} MB "
          f"(resumen {r['MB']} MB) | retr={d.retr.sum()} (resumen {r['retr']})")
    seg.append(d)
seg = pd.concat(seg, ignore_index=True)
orden = seg.groupby("escenario_mbps").puerto_origen.first().rank().astype(int)
seg["orden_ejecucion"] = seg.escenario_mbps.map(orden)
seg.to_csv(BASE / "intervalos_1s.csv", index=False)

# ------------------------------------------------- 2. ventanas de 60 segundos
seg["ventana"] = (seg.t_ini // VENTANA_S).astype(int) + 1
g = seg.groupby(["escenario_mbps", "ventana"])
v = g.agg(
    t_ini_s=("t_ini", "min"), t_fin_s=("t_fin", "max"),
    bytes_tx=("bytes", "sum"), retransmisiones=("retr", "sum"),
    tasa_min_1s_mbps=("mbps", "min"), tasa_max_1s_mbps=("mbps", "max"),
    tasa_std_1s_mbps=("mbps", "std"),
    cwnd_medio_kb=("cwnd_kb", "mean"), cwnd_min_kb=("cwnd_kb", "min"),
    segundos_con_retr=("retr", lambda s: int((s > 0).sum())),
    orden_ejecucion=("orden_ejecucion", "first"),
).reset_index()
v["duracion_s"] = (v.t_fin_s - v.t_ini_s).round(2)
v["replica"] = v["ventana"]  # ventana consecutiva dentro de la misma sesión
v["bytes_rx"] = v["bytes_tx"]  # iperf3 reporta receptor = emisor (0 bytes perdidos)
v["tasa_tx_mbps"] = 8 * v.bytes_tx / (v.duracion_s * 1e6)
v["tasa_rx_mbps"] = 8 * v.bytes_rx / (v.duracion_s * 1e6)
v["segmentos_est"] = v.bytes_tx / MSS
v["retr_por_mil_seg"] = 1000 * v.retransmisiones / v.segmentos_est
v["evento_transitorio"] = (v.tasa_min_1s_mbps < 0.5 * v.escenario_mbps) | (v.cwnd_min_kb < 100)
# Variables que exige el taller pero que NO se midieron en la práctica
for c in ["inicio", "rtt_medio_ms", "perdida_pct", "descartes_tx", "errores_tx"]:
    v[c] = np.nan
v["fuente_bytes"] = "iperf3 (capa de aplicación, intervalos de 1 s)"
v["particion"] = np.where(v.ventana <= 3, "train", "test")
cols = ["inicio", "escenario_mbps", "replica", "orden_ejecucion", "t_ini_s", "t_fin_s",
        "duracion_s", "bytes_rx", "bytes_tx", "tasa_rx_mbps", "tasa_tx_mbps",
        "rtt_medio_ms", "perdida_pct", "descartes_tx", "errores_tx",
        "retransmisiones", "retr_por_mil_seg", "segundos_con_retr",
        "tasa_min_1s_mbps", "tasa_max_1s_mbps", "tasa_std_1s_mbps",
        "cwnd_medio_kb", "cwnd_min_kb", "evento_transitorio", "particion", "fuente_bytes"]
v = v[cols].round(4)
v.to_csv(BASE / "ventanas.csv", index=False)
print("\nventanas.csv:", v.shape)
print(v[["escenario_mbps", "replica", "tasa_tx_mbps", "retransmisiones",
         "retr_por_mil_seg", "tasa_min_1s_mbps", "cwnd_min_kb", "evento_transitorio"]].to_string(index=False))

# --------------------------------------------------- 3. análisis exploratorio
print("\nResumen por escenario (mediana, min, max):")
print(v.groupby("escenario_mbps")[["tasa_tx_mbps", "retransmisiones", "retr_por_mil_seg"]]
      .agg(["median", "min", "max"]).round(3).to_string())

col = {5: "#3b6fb6", 10: "#d9822b", 15: "#2e9e6a"}
fig, axs = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
for obj, d in seg.groupby("escenario_mbps"):
    axs[0].plot(d.t_ini, d.mbps, lw=0.9, color=col[obj], label=f"{obj} Mbps objetivo")
    axs[1].plot(d.t_ini, d.cwnd_kb, lw=0.9, color=col[obj])
    axs[2].plot(d.t_ini, d.retr.rolling(10, min_periods=1).sum(), lw=0.9, color=col[obj])
axs[0].set_ylabel("Tasa 1 s (Mbps)"); axs[1].set_ylabel("Cwnd (KB)")
axs[2].set_ylabel("Retr. (suma móvil 10 s)"); axs[2].set_xlabel("Tiempo de la prueba (s)")
for x in range(60, 300, 60):
    for a in axs: a.axvline(x, color="grey", lw=0.5, ls=":")
axs[0].legend(ncol=3, fontsize=8, loc="upper right"); fig.suptitle("Series de tiempo por escenario (ventanas de 60 s marcadas)")
fig.tight_layout(); fig.savefig(FIG / "fig1_series_tiempo.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
datos_box = [seg[seg.escenario_mbps == o].mbps for o in TRAZAS]
axs[0].boxplot(datos_box, tick_labels=[f"{o}M" for o in TRAZAS])
axs[0].set(title="Tasa por segundo según escenario", ylabel="Mbps")
for o in TRAZAS:
    axs[1].hist(seg[seg.escenario_mbps == o].mbps, bins=40, alpha=0.6, color=col[o], label=f"{o}M")
axs[1].set(title="Histograma de tasa por segundo", xlabel="Mbps", ylabel="Frecuencia"); axs[1].legend()
fig.tight_layout(); fig.savefig(FIG / "fig2_caja_histograma.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
sc = ax.scatter(v.tasa_tx_mbps, v.retransmisiones, c=v.cwnd_min_kb, cmap="viridis", s=60, edgecolor="k")
plt.colorbar(sc, label="Cwnd mínima en la ventana (KB)")
ax.set(xlabel="Tasa TX medida (Mbps)", ylabel="Retransmisiones por ventana de 60 s",
       title="Carga vs. retransmisiones (15 ventanas)")
fig.tight_layout(); fig.savefig(FIG / "fig3_dispersion.png", dpi=150); plt.close(fig)

# ----------------------------------------------- 4. regresión lineal (punto 5)
# Objetivo disponible: retransmisiones por ventana (no hay RTT medido).
OBJ = "retransmisiones"
train = v[v.particion == "train"]; test = v[v.particion == "test"]
X = ["tasa_tx_mbps"]
modelo = LinearRegression().fit(train[X], train[OBJ])
real = test[OBJ].to_numpy(); pred = modelo.predict(test[X])
base = np.full_like(real, train[OBJ].median(), dtype=float)
met = {
    "n_train": len(train), "n_test": len(test),
    "rango_x_train": (train.tasa_tx_mbps.min(), train.tasa_tx_mbps.max()),
    "rango_x_test": (test.tasa_tx_mbps.min(), test.tasa_tx_mbps.max()),
    "b0": modelo.intercept_, "b1": modelo.coef_[0],
    "MAE": mean_absolute_error(real, pred), "RMSE": np.sqrt(mean_squared_error(real, pred)),
    "R2": r2_score(real, pred),
    "MAE_base_mediana": mean_absolute_error(real, base),
    "R2_train": modelo.score(train[X], train[OBJ]),
}
print("\nREGRESIÓN (objetivo = retransmisiones por ventana)")
for k, val in met.items():
    print(f"  {k}: {val}")
r_pearson = np.corrcoef(v.tasa_tx_mbps, v[OBJ])[0, 1]
print(f"  correlación de Pearson (15 ventanas): {r_pearson:.3f}")

# Robustez: validación dejando una sesión (escenario) fuera
print("\nValidación 'dejando un escenario fuera' (extrapolación/interpolación):")
for o in TRAZAS:
    tr, te = v[v.escenario_mbps != o], v[v.escenario_mbps == o]
    m = LinearRegression().fit(tr[X], tr[OBJ])
    print(f"  prueba={o}M  MAE={mean_absolute_error(te[OBJ], m.predict(te[X])):.2f}  "
          f"MAE_mediana={mean_absolute_error(te[OBJ], np.full(len(te), tr[OBJ].median())):.2f}")

test = test.assign(pred=pred, residuo=real - pred)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
lim = [0, max(real.max(), pred.max()) * 1.15]
axs[0].scatter(real, pred, s=60, c=[col[o] for o in test.escenario_mbps], edgecolor="k")
axs[0].plot(lim, lim, "k--", lw=0.8); axs[0].set(xlim=lim, ylim=lim, xlabel="Medido", ylabel="Predicho",
                                                 title="Predicción vs. medición (test)")
axs[1].scatter(test.tasa_tx_mbps, test.residuo, s=60, c=[col[o] for o in test.escenario_mbps], edgecolor="k")
axs[1].axhline(0, color="k", lw=0.8)
axs[1].set(xlabel="Tasa TX (Mbps)", ylabel="Residuo (y - ŷ)", title="Residuos vs. carga (test)")
fig.tight_layout(); fig.savefig(FIG / "fig4_prediccion_residuos.png", dpi=150); plt.close(fig)

# ------------------------------------------ 5. predicción registrada (punto 6)
x_tr = (train.tasa_tx_mbps.min(), train.tasa_tx_mbps.max())
planes = pd.DataFrame({"carga_planificada_mbps": [12.0, 20.0, 25.0, 30.0]})
planes["retr_predichas"] = modelo.predict(planes.rename(columns={"carga_planificada_mbps": "tasa_tx_mbps"}))
planes["tipo"] = np.where(planes.carga_planificada_mbps.between(*x_tr), "interpolación", "extrapolación")
planes["rango_entrenamiento"] = f"{x_tr[0]:.2f}–{x_tr[1]:.2f} Mbps"
planes.round(2).to_csv(BASE / "prediccion_registrada.csv", index=False)
print("\nPredicción registrada antes de medir carga alta:")
print(planes.round(2).to_string(index=False))

pd.Series({k: (str(val) if isinstance(val, tuple) else round(float(val), 4)) for k, val in met.items()}
          ).to_csv(BASE / "metricas_modelo.csv", header=["valor"])
print("\nArchivos generados en", BASE)
