# Wine Quality API 🍷

API REST para predecir la calidad de vinos blancos a partir de características físico-químicas. Implementada en Flask y desplegada en Render.

## Endpoints

### `/` (GET)
Landing page informativa con instrucciones para usar la API.

### `/api/v1/predict` (GET)
Realiza una predicción de calidad del vino.

**Parámetros (query string):**
- `alcohol`: nivel de alcohol
- `pH`: acidez
- `sulphates`: cantidad de sulfatos

**Ejemplo de uso:**
```
https://<añadir-nombre-app!!!>.onrender.com/api/v1/predict?alcohol=10.5&pH=3.2&sulphates=0.5
```

### `/api/v1/retrain` (GET)
Reentrena el modelo con un nuevo dataset (`data/winequality_new.csv`) y guarda el nuevo modelo.

### `/ping` (GET)
Endpoint de prueba para comprobar el funcionamiento de la API o provocar un redeploy.

## Despliegue

### Dependencias
Instalar con:

```
pip install -r requirements.txt
```

### Ejecución local
```
python app_model.py
```

### Despliegue en Render
1. Crear repo en GitHub con esta estructura.
2. En Render:
   - Nuevo servicio web → conectar repo.
   - Comando de arranque:
     ```
     gunicorn app_model:app
     ```

## Dataset

Se utilizan variables del dataset de vinos blancos (`winequality-white.csv`) del UCI ML Repository.

