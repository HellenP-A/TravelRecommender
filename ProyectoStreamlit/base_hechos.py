# Contenido de base_hechos.py
class BaseHechos:
    def __init__(self):
        self.hechos = {}

    def ingresar_datos_usuario(self, presupuesto, duracion_min, duracion_max, mes, tipo_hospedaje):
        self.hechos["presupuesto"] = presupuesto
        self.hechos["duracion_min"] = duracion_min
        self.hechos["duracion_max"] = duracion_max
        self.hechos["mes"] = mes
        self.hechos["tipo_hospedaje"] = tipo_hospedaje if tipo_hospedaje else None

    def obtener_datos_usuario(self):
        return self.hechos