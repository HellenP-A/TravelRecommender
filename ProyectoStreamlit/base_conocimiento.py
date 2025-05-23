# Contenido actualizado de base_conocimiento.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
import Rule as rl
import os

class BaseConocimiento:
    def __init__(self, travel_data):
        self.travel_data = travel_data.copy()
        self.travel_data["Total cost"] = self.travel_data["Accommodation cost"] + self.travel_data["Transportation cost"]
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(
            self.travel_data[["Total cost", "Duration (days)", "Month"]]
        )
        self.rules = self.load_rules()
        print("Reglas cargadas.")
        print("Base de conocimientos inicializada.")

    def load_rules(self):
        # Construir la ruta absoluta al archivo rules.json
        script_dir = os.path.dirname(os.path.abspath(__file__))
        rules_path = os.path.join(script_dir, "rules.json")
        with open(rules_path, "r") as f:
            rules_config = json.load(f)["rules"]
        rules = []
        for rule in rules_config:
            if rule["type"] == "cosine_similarity":
                rules.append(rl.CosineSimilarityRule(self.scaler, self.normalized_data))
            elif rule["type"] == "threshold":
                column = "Total cost" if rule["column"] == "budget" else rule["column"]
                rules.append(rl.ThresholdRule(rule["threshold"], column))
            elif rule["type"] == "equality":
                column = "Accommodation type" if rule["column"] in ["hotel_name", "city_name"] else rule["column"]
                rules.append(rl.EqualityRule(column))
        return rules

    def calcular_similitud(self, presupuesto, duracion_min, duracion_max, mes, tipo_hospedaje=None):
        # Convertir user_input a un DataFrame con los mismos nombres de columnas que se usaron en fit
        user_input_df = pd.DataFrame(
            [[presupuesto, (duracion_min + duracion_max) / 2, mes]],
            columns=["Total cost", "Duration (days)", "Month"]
        )
        user_input_scaled = self.scaler.transform(user_input_df)
        results = []
        for rule in self.rules:
            if isinstance(rule, rl.CosineSimilarityRule):
                result = rule.apply(user_input_scaled[0], self.travel_data)  # Pasamos el array escalado
                results.append(result)
            elif isinstance(rule, rl.ThresholdRule):
                result = rule.apply(presupuesto, self.travel_data)
                if np.isscalar(result):
                    result = np.full(len(self.travel_data), result)
                results.append(result)
            elif isinstance(rule, rl.EqualityRule) and tipo_hospedaje:
                result = rule.apply(tipo_hospedaje, self.travel_data)
                if np.isscalar(result):
                    result = np.full(len(self.travel_data), result)
                results.append(result)
            else:
                results.append(np.zeros(len(self.travel_data)))
        
        results = [np.array(r) for r in results]
        return np.mean(results, axis=0)

    def recomendar_destinos(self, presupuesto, duracion_min, duracion_max, mes, tipo_hospedaje=None):
        self.travel_data["Similarity"] = self.calcular_similitud(presupuesto, duracion_min, duracion_max, mes, tipo_hospedaje)
        destinos_filtrados = self.travel_data.copy()

        destinos_filtrados = destinos_filtrados[destinos_filtrados["Total cost"] <= presupuesto]
        if tipo_hospedaje:
            destinos_filtrados = destinos_filtrados[destinos_filtrados["Accommodation type"] == tipo_hospedaje]
        
        if destinos_filtrados.empty:
            return pd.DataFrame()
        
        destinos_recomendados = destinos_filtrados.sort_values(by="Similarity", ascending=False)
        return destinos_recomendados[["Destination", "Total cost", "Duration (days)", "Accommodation type", "Similarity"]].head(6)
