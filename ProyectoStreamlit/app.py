# Contenido actualizado de app.py
import streamlit as st
import pandas as pd
from motor_inferencia import MotorInferencia
from base_conocimiento import BaseConocimiento
from base_hechos import BaseHechos

# Configuración de la página
st.set_page_config(
    page_title="Recomendador de Destinos Turísticos",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.streamlit.io',
        'Report a Bug': 'https://github.com/streamlit/streamlit/issues',
        'About': 'Sistema de recomendación de destinos turísticos basado en conocimientos - Universidad CENFOTEC'
    }
)

# Cargar dataset
travel_data = pd.read_csv("cleaned_travel_dataset.csv")

# Inicializar sistema
base_conocimiento = BaseConocimiento(travel_data)
base_hechos = BaseHechos()
motor = MotorInferencia(base_conocimiento, base_hechos)

# Barra lateral
st.sidebar.header("📋 Acerca del Proyecto")
st.sidebar.markdown("""
Este proyecto desarrolla un Sistema Basado en Conocimientos (KBS) para la planificación de vacaciones, abordando las problemáticas de la falta de información consolidada, la búsqueda manual ineficiente y la escasa personalización en las recomendaciones de viaje.  
Utilizamos el dataset *Travel Details* de Kaggle para ofrecer recomendaciones optimizadas según presupuesto, duración y preferencias de hospedaje.
""")

st.sidebar.markdown("### 🛠️ Detalles del Desarrollo")
st.sidebar.markdown("""
- **Problemática:** Falta de información consolidada, necesidad de comparar hospedaje y transporte, búsqueda manual ineficiente, y escasa personalización.  
- **Requerimientos:**  
  - Interfaz de usuario interactiva para ingreso de datos.  
  - Base de Conocimientos con reglas sobre destinos, costos y condiciones.  
  - Base de Hechos para almacenar datos del usuario.  
  - Motor de Inferencia con similitud de coseno y reglas de filtrado.  
  - Uso exclusivo de datos de Kaggle, sin APIs externas.  
- **Arquitectura:**  
  - Frontend: Interfaz web con Streamlit (adaptada desde un CLI inicial).  
  - Backend: Python con Pandas para manipulación de datos.  
  - Base de Conocimientos: Estructurada en CSV (adaptada desde OWL/RDF).  
  - Base de Hechos: Almacenamiento en memoria (adaptada desde MongoDB/PostgreSQL).  
  - Motor de Inferencia: Implementado con reglas personalizadas (adaptado desde PyKE/RDFLib).  
""")

st.sidebar.markdown("### 📚 Lecciones Aprendidas")
st.sidebar.markdown("""
- **Preprocesamiento:** La limpieza del dataset fue clave para garantizar recomendaciones precisas.  
- **Similitud de coseno:** Efectiva para recomendaciones flexibles, evitando filtros estrictos.  
- **Reglas:** La combinación de umbrales y preferencias mejora la personalización.  
- **Interfaz:** Streamlit permitió un desarrollo rápido y una experiencia de usuario intuitiva.  
- **Limitaciones:** La falta de datos dinámicos (como clima o popularidad) podría enriquecer las recomendaciones.
""")

st.sidebar.markdown("---")  # Línea divisoria para mejor separación visual
st.sidebar.markdown("Desarrollado para el curso de **Sistemas Basados en Conocimientos** de la **Universidad CENFOTEC**.")
st.sidebar.markdown("**Profesor:** Luis Gerardo León Vega")
st.sidebar.markdown("### 👥 Autores")
st.sidebar.markdown("""
<div style='background-color: #1E1E1E; padding: 10px; border-radius: 5px;'>
  <p style='margin: 5px 0; color: #FAFAFA;'><span style='color: #FF4B4B;'>👤</span> Hellen Aguilar Noguera</p>
  <p style='margin: 5px 0; color: #FAFAFA;'><span style='color: #FF4B4B;'>👤</span> José Leonardo Araya Parajeles</p>
  <p style='margin: 5px 0; color: #FAFAFA;'><span style='color: #FF4B4B;'>👤</span> Fernando Rojas Meléndez</p>
  <p style='margin: 5px 0; color: #FAFAFA;'><span style='color: #FF4B4B;'>👤</span> Alejandro Villalobos Hernández</p>
</div>
""", unsafe_allow_html=True)

# Encabezado personalizado (sin redundancia)
st.markdown("""
# Sistema de Recomendación de Destinos Turísticos
""")

# Mostrar la imagen (si ya la tienes integrada)
# try:
#     st.image("travel_illustration.jpg", use_column_width=True)
# except FileNotFoundError:
#     st.warning("La imagen 'travel_illustration.jpg' no se encontró. Asegúrate de cargarla en el contenedor.")

# Estilo visual personalizado (tema oscuro)
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;  /* Fondo oscuro */
}
h1 {
    color: #FF4B4B;  /* Título en rojo claro */
}
h2, h3 {
    color: #FAFAFA;  /* Subtítulos en blanco */
}
.stButton>button {
    background-color: #FF4B4B;  /* Botón en rojo claro */
    color: white;  /* Texto del botón en blanco */
}
.stButton>button:hover {
    background-color: #E43A3A;  /* Color del botón al pasar el mouse */
}
.stSidebar .sidebar-content {
    background-color: #262730;  /* Fondo de la barra lateral */
}
</style>
""", unsafe_allow_html=True)

# Contenido principal
st.title("✈️ Recomendador de Destinos Turísticos")
st.write("Utilice el formulario a continuación para obtener recomendaciones personalizadas de destinos turísticos.")

# Formulario para ingresar datos
st.header("Ingrese sus Preferencias de Viaje")

with st.form("travel_form"):
    # Subsección: Presupuesto
    st.subheader("💰 Presupuesto")
    presupuesto = st.number_input(
        "Presupuesto (USD)",
        min_value=100.0,
        step=50.0,
        value=1000.0,
        help="Ingrese su presupuesto máximo en dólares para el viaje."
    )

    # Subsección: Duración del Viaje
    st.subheader("⏳ Duración del Viaje")
    col_dur1, col_dur2 = st.columns(2)
    with col_dur1:
        duracion_min = st.number_input(
            "Duración mínima (días)",
            min_value=1,
            step=1,
            value=3,
            help="Número mínimo de días que planea viajar."
        )
    with col_dur2:
        duracion_max = st.number_input(
            "Duración máxima (días)",
            min_value=1,
            step=1,
            value=7,
            help="Número máximo de días que planea viajar."
        )

    # Subsección: Fecha del Viaje
    st.subheader("📅 Fecha del Viaje")
    mes = st.slider(
        "Mes de viaje",
        1, 12, 6,
        help="Seleccione el mes en el que desea viajar (1: Enero, 12: Diciembre)."
    )

    # Subsección: Preferencias de Hospedaje
    st.subheader("🏨 Preferencias de Hospedaje")
    tipo_hospedaje = st.selectbox(
        "Tipo de hospedaje",
        ["", "Hotel", "Resort", "Villa", "Airbnb"],
        index=0,
        help="Seleccione el tipo de hospedaje que prefiere. Deje en blanco para no filtrar por este criterio."
    )

    # Botón para realizar la búsqueda
    submitted = st.form_submit_button("Buscar Destinos")

if submitted:
    # Ingresar datos del usuario en la BaseHechos
    base_hechos.ingresar_datos_usuario(presupuesto, duracion_min, duracion_max, mes, tipo_hospedaje)
    
    # Generar recomendaciones
    recomendaciones = motor.generar_recomendaciones()

    # Mostrar el resultado
    st.subheader("Resultado de las Recomendaciones")

    if not recomendaciones.empty:
        st.success("🎯 **Recomendaciones encontradas:**")
        st.dataframe(recomendaciones.style.format({"Total cost": "${:.2f}", "Similarity": "{:.4f}"}))
    else:
        st.warning("😕 **No se encontraron coincidencias con tus criterios.** Intenta ajustar tus preferencias.")

# Pie de página
st.markdown("""
---
**© 2025 Hellen Aguilar Noguera, José Leonardo Araya Parajeles, Fernando Rojas Meléndez, Alejandro Villalobos Hernández**  
Desarrollado para el curso de Sistemas Basados en Conocimientos, Universidad CENFOTEC.
""", unsafe_allow_html=True)