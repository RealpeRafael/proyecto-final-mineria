# Proyecto Final - Minería de Datos

Descripción
-----------
Repositorio del proyecto final de Minería de Datos: preparación de datos, ingeniería de características, entrenamiento de modelos y una aplicación de demostración en Streamlit para predecir la duración de la estadía hospitalaria.

Estructura del repositorio
--------------------------
- `app/` - Código de la aplicación y transformadores. Contiene `app.py` (generador de la app Streamlit) y `transformadores.py` con clases de preprocesamiento (LimpiezaInicial, ImputacionNulos, AgrupacionCCS).
- `data/raw/` - Datos originales y documentación del dataset.
- `data/processed/` - Datos procesados listos para modelado (ej. `muestra_sparcs_10000.csv`).
- `models/` - Artefactos de modelos y pipelines entrenados (colocar `modelo_final_pipeline.pkl` aquí o en la raíz según la configuración).
- `notebooks/` - Notebooks de exploración y experimentación (`PreparacionDatos.ipynb`, `Modelos.ipynb`).
- `requirements.txt` - Dependencias del proyecto.

Requisitos
----------
Instalar dependencias (recomendado en un entorno virtual):

```bash
python -m pip install -r requirements.txt
```

Uso
---
1. Colocar el artefacto `modelo_final_pipeline.pkl` en la raíz del proyecto o en la ruta esperada por la aplicación.
2. Ejecutar la aplicación Streamlit localmente:

```bash
streamlit run app/app.py
```

Otras pruebas y scripts
-----------------------
- `app/transformadores.py` contiene transformadores personalizados usados en el pipeline (limpieza, imputación y agrupación de códigos CCS).
- Los notebooks en `notebooks/` muestran la preparación de datos y el entrenamiento de modelos.

Autores
-------
Rafael Gutiérrez, Sara Rúa y Valentina Leal fuimos los creadores de este proyecto.

Licencia
--------
El repositorio no especifica una licencia; añada un archivo `LICENSE` si desea aclarar términos de uso.

Contacto
-------
Para preguntas o colaboración, abrir un issue en este repositorio o contactar a los autores.
