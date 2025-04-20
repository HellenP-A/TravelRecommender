import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from motor_inferencia import MotorInferencia
from base_conocimiento import BaseConocimiento
from base_hechos import BaseHechos

# 1) Carga y setup
df_data = pd.read_csv("cleaned_travel_dataset.csv")
bc = BaseConocimiento(df_data)
bh = BaseHechos()
motor = MotorInferencia(bc, bh)

# 2) Define tu test set con ground truth
test_set = [
    {
        "name": "Escenario hotel barato",
        "presupuesto": 1000, "dur_min": 3, "dur_max": 5,
        "mes": 6, "tipo_hosp": "Hotel",
        "relevantes": ["Paris, France", "London, UK"]
    },
    {
        "name": "Viaje largo en resort",
        "presupuesto": 8000, "dur_min": 14, "dur_max": 21,
        "mes": 12, "tipo_hosp": "Resort",
        "relevantes": ["Bali, Indonesia", "Phuket, Thailand"]
    },
    {
        "name": "Escapada sin hospedaje",
        "presupuesto": 2000, "dur_min": 2, "dur_max": 4,
        "mes": 4, "tipo_hosp": None,
        "relevantes": ["London, UK", "New York, USA"]
    },
    {
        "name": "Vacaciones en Villa lujosa",
        "presupuesto": 15000, "dur_min": 7, "dur_max": 10,
        "mes": 8, "tipo_hosp": "Villa",
        "relevantes": ["Bali, Indonesia", "Phuket, Thailand"]
    },
    {
        "name": "Semana estándar en hotel",
        "presupuesto": 3000, "dur_min": 5, "dur_max": 7,
        "mes": 5, "tipo_hosp": "Hotel",
        "relevantes": ["New York, USA", "Paris, France"]
    },
    {
        "name": "Escapada otoñal sin filtro",
        "presupuesto": 2500, "dur_min": 4, "dur_max": 6,
        "mes": 10, "tipo_hosp": None,
        "relevantes": ["New York, USA", "Paris, France"]
    },
    {
        "name": "Viaje familiar en resort",
        "presupuesto": 6000, "dur_min": 10, "dur_max": 14,
        "mes": 7, "tipo_hosp": "Resort",
        "relevantes": ["Phuket, Thailand", "Bali, Indonesia"]
    },
    {
        "name": "Escapada febrero low‑cost",
        "presupuesto": 800, "dur_min": 3, "dur_max": 5,
        "mes": 2, "tipo_hosp": None,
        "relevantes": ["London, UK", "Paris, France"]
    },
]


K = 6  # top‑K

results = []
for case in test_set:
    # inyecta hechos
    bh.ingresar_datos_usuario(
        case["presupuesto"],
        case["dur_min"],
        case["dur_max"],
        case["mes"],
        case["tipo_hosp"]
    )
    # genera top‑K
    recs = motor.generar_recomendaciones()
    preds = recs["Destination"].tolist()[:K]

    # 3) construye vectores binarios sobre TODO el catálogo
    all_dest = bc.travel_data["Destination"].tolist()
    y_true = [1 if d in case["relevantes"] else 0 for d in all_dest]
    y_pred = [1 if d in preds            else 0 for d in all_dest]

    # métricas de clasificación
    acc = (confusion_matrix(y_true, y_pred).trace() /
           sum(confusion_matrix(y_true, y_pred).ravel()))
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    results.append({
        "Caso":      case["name"],
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "TP":        cm[1,1],
        "FP":        cm[0,1],
        "FN":        cm[1,0],
        "TN":        cm[0,0],
    })

# 4) muestra resultados
df_metrics = pd.DataFrame(results)
print(df_metrics.to_markdown(index=False))
