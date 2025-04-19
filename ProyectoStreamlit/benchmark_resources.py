#!/usr/bin/env python3
import os
import time
import csv

import pandas as pd
import psutil                        # pip install psutil
from motor_inferencia import MotorInferencia
from base_conocimiento import BaseConocimiento
from base_hechos import BaseHechos

# --- Carga robusta del CSV (igual que antes) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
paths = [
    os.path.join(script_dir, "cleaned_travel_dataset.csv"),
    os.path.join(script_dir, "ProyectoStreamlit", "cleaned_travel_dataset.csv")
]
for p in paths:
    if os.path.exists(p):
        travel_data = pd.read_csv(p)
        break
else:
    raise FileNotFoundError(f"No se encontró cleaned_travel_dataset.csv en {paths}")

# --- Inicialización ---
base_conocimiento = BaseConocimiento(travel_data)
base_hechos      = BaseHechos()
motor            = MotorInferencia(base_conocimiento, base_hechos)
proc             = psutil.Process(os.getpid())

# --- Parámetros ---
N = 30  # muestras por máquina
case = {"name": "Medición global", "presupuesto":1000, "dur_min":3, "dur_max":5, "mes":6, "tipo_hosp": None}

# --- Prepara CSV de salida ---
with open("resource_benchmark.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "run","cpu_percent","mem_rss_mb",
        "disk_read_bytes","disk_write_bytes",
        "net_sent_bytes","net_recv_bytes"
    ])
    writer.writeheader()

    base_hechos.ingresar_datos_usuario(
        case["presupuesto"], case["dur_min"],
        case["dur_max"], case["mes"], case["tipo_hosp"]
    )
    # Inicializa contadores base
    for i in range(1, N+1):
        # Tomar snapshot inicial
        t0_cpu = proc.cpu_times()
        d0     = psutil.disk_io_counters()
        n0     = psutil.net_io_counters()
        t0     = time.time()

        # Llamada a inferencia
        motor.generar_recomendaciones()

        # Tomar snapshot final
        t1     = time.time()
        t1_cpu = proc.cpu_times()
        d1     = psutil.disk_io_counters()
        n1     = psutil.net_io_counters()

        # Calcular diferencias
        cpu_time_delta = (t1_cpu.user + t1_cpu.system) - (t0_cpu.user + t0_cpu.system)
        cpu_percent    = cpu_time_delta / (t1 - t0) * 100
        mem_rss_mb     = proc.memory_info().rss / (1024**2)
        disk_read      = d1.read_bytes - d0.read_bytes
        disk_write     = d1.write_bytes - d0.write_bytes
        net_sent       = n1.bytes_sent - n0.bytes_sent
        net_recv       = n1.bytes_recv - n0.bytes_recv

        # Guardar fila
        writer.writerow({
            "run": i,
            "cpu_percent": f"{cpu_percent:.2f}",
            "mem_rss_mb": f"{mem_rss_mb:.2f}",
            "disk_read_bytes": disk_read,
            "disk_write_bytes": disk_write,
            "net_sent_bytes": net_sent,
            "net_recv_bytes": net_recv
        })

print("✅ Se recopiló resource_benchmark.csv con 30 muestras.")

# Cargar los datos de las 30 corridas
df = pd.read_csv("resource_benchmark.csv")

# Calcular promedio de cada columna numérica
summary = df.mean(numeric_only=True)

print("📊 Promedio de las 30 corridas:")
print(summary.to_frame(name="promedio").T)

# (Opcional) Guardar el resumen a un CSV
summary.to_frame(name="promedio").T.to_csv("resource_summary.csv", index=False)
print("▶️ Resumen guardado en resource_summary.csv")
