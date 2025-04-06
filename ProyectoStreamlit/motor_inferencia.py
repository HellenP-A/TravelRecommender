# Contenido actualizado de motor_inferencia.py
class MotorInferencia:
    def __init__(self, base_conocimiento, base_hechos):
        self.base_conocimiento = base_conocimiento
        self.base_hechos = base_hechos

    def generar_recomendaciones(self):
        datos_usuario = self.base_hechos.obtener_datos_usuario()
        recomendaciones = self.base_conocimiento.recomendar_destinos(
            datos_usuario["presupuesto"],
            datos_usuario["duracion_min"],
            datos_usuario["duracion_max"],
            datos_usuario["mes"],
            datos_usuario["tipo_hospedaje"]
        )
        return recomendaciones.head(6)