# Contenido de Rule.py
from sklearn.metrics.pairwise import cosine_similarity

class Rule:
    """Clase base para todas las reglas."""
    def apply(self, user_input, travel_data):
        raise NotImplementedError("Cada regla debe implementar 'apply'")

class CosineSimilarityRule(Rule):
    def __init__(self, scaler, normalized_data):
        self.scaler = scaler
        self.normalized_data = normalized_data

    def apply(self, user_input, travel_data):
        user_input_scaled = self.scaler.transform([user_input])
        return cosine_similarity(user_input_scaled, self.normalized_data)[0]

class ThresholdRule(Rule):
    def __init__(self, threshold, column_name):
        self.threshold = threshold
        self.column_name = column_name

    def apply(self, user_input, travel_data):
        if self.column_name not in travel_data.columns:
            return 0  # Retorna 0 si la columna no existe
        return (travel_data[self.column_name] <= self.threshold).astype(float)

class EqualityRule(Rule):
    def __init__(self, column_name):
        self.column_name = column_name

    def apply(self, user_input, travel_data):
        if self.column_name not in travel_data.columns or user_input is None:
            return 0  # Retorna 0 si no hay coincidencia posible
        # Aquí asumimos que user_input es una lista y el valor relevante está en una posición específica
        # Para este caso, usaremos el tipo_hospedaje como entrada adicional si está presente
        return (travel_data[self.column_name] == user_input).astype(float)