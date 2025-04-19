#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd

from ProyectoStreamlit.base_conocimiento import BaseConocimiento
from ProyectoStreamlit.base_hechos import BaseHechos
from ProyectoStreamlit.motor_inferencia import MotorInferencia

# --- Carga robusta del CSV ---
script_dir = os.path.dirname(os.path.abspath(__file__))
paths = [
    os.path.join(script_dir, "cleaned_travel_dataset.csv"),
    os.path.join(script_dir, "ProyectoStreamlit", "cleaned_travel_dataset.csv")
]

for p in paths:
    if os.path.exists(p):
        travel_data = pd.read_csv(p)
        print(f"Cargado dataset desde: {p}")
        break
else:
    raise FileNotFoundError(f"No se encontró cleaned_travel_dataset.csv en {paths}")
base_conocimiento = BaseConocimiento(travel_data)
base_hechos = BaseHechos()
motor = MotorInferencia(base_conocimiento, base_hechos)

# --- Definición de casos de prueba (total 13 escenarios) ---
test_cases = [
    {"name": "Escenario hotel barato",
     "presupuesto": 1000, "dur_min": 3, "dur_max": 5, "mes": 6, "tipo_hosp": "Hotel"},
    {"name": "Viaje largo en resort",
     "presupuesto": 8000, "dur_min": 14, "dur_max": 21, "mes": 12, "tipo_hosp": "Resort"},
    {"name": "Escapada sin hospedaje",
     "presupuesto": 2000, "dur_min": 2, "dur_max": 4, "mes": 4, "tipo_hosp": None},
    # Escenarios adicionales
    {"name": "Fin de semana en Airbnb barato",
     "presupuesto": 500,  "dur_min": 1,  "dur_max": 2,  "mes": 3,  "tipo_hosp": "Airbnb"},
    {"name": "Vacaciones en Villa lujosa",
     "presupuesto": 15000,"dur_min": 7,  "dur_max": 10, "mes": 8,  "tipo_hosp": "Villa"},
    {"name": "Semana estándar en hotel",
     "presupuesto": 3000, "dur_min": 5,  "dur_max": 7,  "mes": 5,  "tipo_hosp": "Hotel"},
    {"name": "Escapada otoñal sin filtro",
     "presupuesto": 2500, "dur_min": 4,  "dur_max": 6,  "mes": 10, "tipo_hosp": None},
    {"name": "Viaje familiar en resort",
     "presupuesto": 6000, "dur_min": 10, "dur_max": 14, "mes": 7,  "tipo_hosp": "Resort"},
    {"name": "Escapada febrero low‑cost",
     "presupuesto": 800,  "dur_min": 3,  "dur_max": 5,  "mes": 2,  "tipo_hosp": None},
    {"name": "Vacaciones de invierno en hotel",
     "presupuesto": 12000,"dur_min": 10, "dur_max": 15, "mes": 1,  "tipo_hosp": "Hotel"},
    {"name": "Viaje de trabajo en Airbnb",
     "presupuesto": 1800, "dur_min": 2,  "dur_max": 4,  "mes": 11, "tipo_hosp": "Airbnb"},
    {"name": "Retiro de fin de año en Villa",
     "presupuesto": 20000,"dur_min": 14, "dur_max": 21, "mes": 12, "tipo_hosp": "Villa"},
{"name": "Lujo en villa verano",
     "presupuesto": 15000, "dur_min": 7,  "dur_max": 14, "mes": 7,  "tipo_hosp": "Villa"},
    {"name": "Vacaciones familiares Airbnb",
     "presupuesto": 3500, "dur_min": 5,  "dur_max": 10, "mes": 7,  "tipo_hosp": "Airbnb"},
    {"name": "Escapada rápida en octubre",
     "presupuesto": 1500, "dur_min": 1,  "dur_max": 3,  "mes": 10, "tipo_hosp": None},
    {"name": "Viaje cultural en invierno",
     "presupuesto": 3000, "dur_min": 5,  "dur_max": 10, "mes": 1,  "tipo_hosp": "Hotel"},
    {"name": "Retiro ecológico primavera",
     "presupuesto": 2500, "dur_min": 7,  "dur_max": 14, "mes": 5,  "tipo_hosp": "Resort"},
    {"name": "Fin de semana exprés",
     "presupuesto": 800,  "dur_min": 2,  "dur_max": 3,  "mes": 9,  "tipo_hosp": "Airbnb"},
    {"name": "Estudio de idiomas otoño",
     "presupuesto": 5000, "dur_min": 30, "dur_max": 60, "mes": 9,  "tipo_hosp": "Hotel"},
    {"name": "Crucero caribeño",
     "presupuesto": 12000,"dur_min": 5,  "dur_max": 10, "mes": 3,  "tipo_hosp": "Resort"},
    {"name": "Aventura mochilero",
     "presupuesto": 1000, "dur_min": 10, "dur_max": 20, "mes": 2,  "tipo_hosp": None},
    {"name": "Vacaciones extremas",
     "presupuesto": 4000, "dur_min": 1,  "dur_max": 1,  "mes": 2,  "tipo_hosp": "Airbnb"},
{"name": "Vacaciones familiares en Airbnb",
     "presupuesto": 3500, "dur_min": 5, "dur_max": 10, "mes": 7, "tipo_hosp": "Airbnb"},
    {"name": "Escapada de fin de semana",
     "presupuesto": 500,  "dur_min": 1, "dur_max": 2, "mes": 3, "tipo_hosp": None},
    {"name": "Vacaciones de lujo en villa",
     "presupuesto": 20000,"dur_min": 7, "dur_max": 14,"mes": 8, "tipo_hosp": "Villa"},
    {"name": "Viaje de negocios estándar",
     "presupuesto": 1500, "dur_min": 3, "dur_max": 5, "mes": 11, "tipo_hosp": "Hotel"},
    {"name": "Tour extendido sin filtro",
     "presupuesto": 6000, "dur_min": 10,"dur_max": 20,"mes": 5, "tipo_hosp": None},
    {"name": "Vacaciones económicas largas",
     "presupuesto": 1200, "dur_min": 14,"dur_max": 21,"mes": 9, "tipo_hosp": None},
    {"name": "Temporada alta en resort",
     "presupuesto": 10000,"dur_min": 7, "dur_max": 10,"mes": 12,"tipo_hosp": "Resort"},
    {"name": "Escapada precio medio",
     "presupuesto": 2500, "dur_min": 4, "dur_max": 6, "mes": 10, "tipo_hosp": "Hotel"},
    {"name": "Primavera en hotel",
     "presupuesto": 3000, "dur_min": 5, "dur_max": 8, "mes": 4, "tipo_hosp": "Hotel"},
    {"name": "Otoño en villa",
     "presupuesto": 8000, "dur_min": 6, "dur_max": 12, "mes": 10, "tipo_hosp": "Villa"}
]

# --- Medición ---
N = 30  # muestras por caso
records = []

for case in test_cases:
    base_hechos.ingresar_datos_usuario(
        case["presupuesto"],
        case["dur_min"],
        case["dur_max"],
        case["mes"],
        case["tipo_hosp"]
    )
    for run in range(1, N + 1):
        t0 = time.time()
        motor.generar_recomendaciones()
        t1 = time.time()
        records.append({
            "case": case["name"],
            "run": run,
            "time_s": t1 - t0
        })

# --- Crear DataFrame ---
df = pd.DataFrame(records)
df.to_csv("benchmark_results.csv", index=False)
print("▶️ Todos los tiempos de inferencia guardados en benchmark_results.csv")

# --- Resumen por caso ---
summary = (
    df
    .groupby("case")["time_s"]
    .agg(tiempo_promedio_s="mean", desviacion_estd_s="std")
)
summary["inferencias_por_s"] = 1 / summary["tiempo_promedio_s"]

# Calculamos mean y std sobre TODAS las muestras:
prom_gen = df["time_s"].agg(["mean", "std"])
# Renombramos índices para que coincidan:
prom_gen.index = ["tiempo_promedio_s", "desviacion_estd_s"]
# Añadimos inferencias por segundo:
prom_gen["inferencias_por_s"] = 1 / prom_gen["tiempo_promedio_s"]

# Insertamos como una nueva fila en el resumen:
summary.loc["Promedio general"] = prom_gen

# --- Exportar resumen completo ---
summary.to_csv("benchmark_summary.csv")
print("▶️ Resumen por caso (y promedio global) guardado en benchmark_summary.csv")