# Taller 5 — Análisis de tráfico LAN y predicción bajo congestión

## Contenido
| Archivo | Descripción |
| --- | --- |
| `datos/traza5.txt`, `traza10.txt`, `traza15.txt` | Salidas originales de iperf3 (TCP, 300 s, -b 5M/10M/15M) |
| `analisis_taller5.py` | Script único: lectura, ventanas, exploración, regresión y predicción |
| `ventanas.csv` | 15 ventanas de 60 s (una fila por ventana) |
| `intervalos_1s.csv` | 900 intervalos de 1 s extraídos de las trazas |
| `metricas_modelo.csv` | Coeficientes, MAE, RMSE, R² y línea base |
| `prediccion_registrada.csv` | Predicción para cargas planificadas (punto 6) |
| `figuras/` | Figuras 1–4 del informe |

## Ejecución
```bash
pip install pandas numpy matplotlib scikit-learn
python analisis_taller5.py
```
Se ejecuta desde esta carpeta; regenera todos los CSV y figuras.

## Versiones usadas
Python 3.13 · pandas 3.0.5 · scikit-learn 1.9.1 · matplotlib 3.11.2 · iperf3 (cliente/servidor)

## Montaje experimental
- Red Wi-Fi de un celular (punto de acceso). Servidor B 10.124.30.91 (`iperf3 -s`), cliente A 10.124.30.129.
- `iperf3 -c 10.124.30.91 -b {5M,10M,15M} -t 300`, ejecutados en ese orden (deducido del puerto origen).

## Qué fue medido y qué no
- **Medido:** bytes por segundo, tasa, retransmisiones TCP y Cwnd (reporte de iperf3).
- **No medido:** RTT y pérdida (ping), contadores de interfaz (`ip -s link`), captura `tcpdump`, hora de inicio.
  Esas columnas quedan vacías en `ventanas.csv`.
- Por falta de RTT, la regresión usa `retransmisiones` como objetivo. Para usar RTT: agregar la columna
  `rtt_medio_ms` a `ventanas.csv` y cambiar `OBJ = "rtt_medio_ms"` en el script.
