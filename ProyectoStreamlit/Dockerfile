# Contenido de Dockerfile
FROM python:3.11-slim

WORKDIR /usr/local/sbec

COPY ./__init__.py ./
COPY ./Rule.py ./
COPY ./rules.json ./
COPY ./cleaned_travel_dataset.csv ./
COPY ./requirements.txt ./
COPY ./app.py ./
COPY ./base_conocimiento.py ./
COPY ./base_hechos.py ./
COPY ./motor_inferencia.py ./

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]