#!/usr/bin/env python3
import os
import time
import threading
import psutil
import numpy as np
import pandas as pd
from motor_inferencia import MotorInferencia
from base_conocimiento import BaseConocimiento
from base_hechos import BaseHechos

# === Carga del dataset ===
script_dir = os.path.dirname(os.path.abspath(__file__))
paths = [
    os.path.join(script_dir, "cleaned_travel_dataset.csv"),
    os.path.join(script_dir, "ProyectoStreamlit", "cleaned_travel_dataset.csv"),
]
for p in paths:
    if os.path.exists(p):
        travel_data = pd.read_csv(p)
        break
else:
    raise FileNotFoundError(f"No cleaned_travel_dataset.csv en {paths}")

# === Inicialización del sistema ===
base_conocimiento = BaseConocimiento(travel_data)
base_hechos      = BaseHechos()
motor            = MotorInferencia(base_conocimiento, base_hechos)

# === Parámetros ===
TDP_W           = 20.0    # vatios estimados para M2
N_MUESTRAS      = 30      # número de mediciones
SAMPLE_INTERVAL = 0.1     # s, muestreo de CPU
BATCH_SIZE      = 1000    # inferencias por medición

# Datos de usuario (ajusta según tu caso real)
PRESUPUESTO, DUR_MIN, DUR_MAX, MES, TIPO_HOSP = 1000, 3, 7, 6, None

def batch_inference():
    """Ejecuta BATCH_SIZE inferencias seguidas."""
    base_hechos.ingresar_datos_usuario(PRESUPUESTO, DUR_MIN, DUR_MAX, MES, TIPO_HOSP)
    for _ in range(BATCH_SIZE):
        motor.generar_recomendaciones()

def measure_energy():
    """Mide runtime, uso de CPU y energía para BATCH_SIZE inferencias."""
    cpu_samples = []
    thread = threading.Thread(target=batch_inference)
    start_time = time.time()
    thread.start()
    while thread.is_alive():
        cpu_samples.append(psutil.cpu_percent(interval=SAMPLE_INTERVAL))
    runtime = time.time() - start_time            # tiempo total batch (s)
    avg_cpu = np.mean(cpu_samples) if cpu_samples else 0.0

    avg_power_w = (avg_cpu / 100.0) * TDP_W        # vatios
    energy_j    = avg_power_w * runtime           # julios
    energy_kwh  = energy_j / 3_600_000            # kWh

    # Energía por inferencia:
    energy_j_per = energy_j / BATCH_SIZE
    energy_kwh_per = energy_kwh / BATCH_SIZE

    return runtime, avg_cpu, energy_j_per, energy_kwh_per

if __name__ == "__main__":
    results = [measure_energy() for _ in range(N_MUESTRAS)]
    runtimes, cpus, joules_per, kwhs_per = zip(*results)

    print(f"\n=== Energía (por inferencia) en {N_MUESTRAS} muestras de {BATCH_SIZE} inf. cada una ===")
    print(f"Tiempo batch medio      : {np.mean(runtimes):.3f} s  ± {np.std(runtimes):.3f}")
    print(f"Uso CPU medio           : {np.mean(cpus):.1f}%    ± {np.std(cpus):.1f}")
    print(f"Energía por inferencia  : {np.mean(joules_per):.6f} J  ± {np.std(joules_per):.6f}")
    print(f"Energía por inferencia  : {np.mean(kwhs_per):.9f} kWh ± {np.std(kwhs_per):.9f}")
