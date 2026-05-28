# Proyecto Final - Minería de Datos: Predicción de Estancia Hospitalaria (SPARCS)

## Descripción
Este repositorio contiene el desarrollo del proyecto final de Minería de Datos enfocado en la predicción de la duración de la estadía hospitalaria utilizando el dataset **SPARCS 2015 (Statewide Planning and Research Cooperative System)** del Departamento de Salud del Estado de Nueva York. 

Siguiendo la metodología **CRISP-DM**, el proyecto abarca desde el entendimiento del negocio, la limpieza profunda y el tratamiento del desbalanceo categórico mediante `SMOTEN`, hasta el entrenamiento, optimización y puesta en producción de modelos predictivos mediante una aplicación web interactiva en **Streamlit**.

---

## Estructura del Repositorio
- `app/` - Código de la aplicación productiva. Contiene `app.py` (interfaz en Streamlit) y los scripts de soporte.
- `data/raw/` - Datos originales y documentación del dataset de salud.
- `data/processed/` - Matriz de datos procesados tras ingeniería de características y dummies.
- `models/` - Contiene `artefacto_final.pkl`, el archivo binario unificado que encapsula el pipeline de preprocesamiento, la codificación *One-Hot Encoding* y el estimador optimizado.
- `notebooks/` - Notebooks y scripts de experimentación y entrenamiento (`PreparacionDatos.ipynb`, `modelos (1).py`).
- `requirements.txt` - Lista de dependencias del proyecto.

---

## El Modelo Predictivo y Soporte Estadístico

A través de un riguroso análisis estadístico mediante pruebas de Shapiro-Wilk, Levene y un **ANOVA de un factor complementado con la prueba post-hoc de Tukey HSD**, se determinó de manera concluyente que la Regresión Logística (LR), la Support Vector Machine (SVM) y el Random Forest (RF) forman el conjunto de modelos con el rendimiento más alto y estadísticamente indistinguibles entre sí para este conjunto de datos ($7,696 \times 382$ registros en entrenamiento).

### Selección Final: SVM (Support Vector Machine)
Se seleccionó la **SVM** refinada mediante búsqueda de hiperparámetros debido a que obtuvo el mayor **F1-Score** en validación cruzada. En el contexto de la gestión hospitalaria de SPARCS, aunque la diferencia numérica con respecto a LR o RF sea menor al 2%, predecir correctamente hospitalizaciones prolongadas adicionales permite una planificación presupuestal óptima y mitiga el riesgo financiero derivado de la saturación de camas y recursos médicos.

### Nota sobre el Tiempo Computacional
El proceso de optimización de hiperparámetros requirió un uso intensivo de la CPU virtual en Google Colab (`n_jobs=-1`). Mientras que las búsquedas por grilla (`GridSearchCV`) de LR y RF tomaron el tiempo estándar, **la optimización bayesiana (`BayesSearchCV`) aplicada sobre la SVM demandó aproximadamente 30 minutos continuos de ejecución** debido a su naturaleza intrínsecamente secuencial, quedando registrado como la principal lección aprendida sobre costo-beneficio de cómputo en el proyecto.

---

## Mitigación de Fuga de Información (*Data Leakage*)
Para garantizar la aplicabilidad real del modelo en el momento del ingreso del paciente, se excluyeron categóricamente variables que se generan únicamente al momento del egreso hospitalario (como `Total Charges`, `Total Costs` y `Patient Disposition`). Su inclusión habría provocado un modelo artificialmente perfecto en el entrenamiento, pero completamente inútil en el entorno de admisión real.

---

## Requisitos
Instalar las dependencias del ecosistema de Machine Learning y la interfaz web (se recomienda usar un entorno virtual):

```bash
python -m pip install -r requirements.txt
Las librerías core del proyecto incluyen: scikit-learn, imbalanced-learn (para el algoritmo SMOTEN), scikit-optimize y streamlit.

Uso y Despliegue de la Aplicación
1. Preparación del Artefacto
Asegúrese de que el archivo binario exportado artefacto_final.pkl se encuentre en la carpeta models/ o en la raíz del proyecto según lo mapeado en el script. Este artefacto es cargado en memoria de forma eficiente una sola vez al inicializar la aplicación.

2. Ejecutar la Aplicación Streamlit
Para levantar la plataforma interactiva localmente ejecute:

Bash
streamlit run app/app.py
3. Funcionamiento de la Interfaz
La aplicación desplegará un formulario web interactivo donde el personal hospitalario introduce los datos de admisión del paciente a través de menús desplegables dinámicos (st.selectbox). Al accionar el botón de predicción, el sistema evalúa los datos a través del pipeline en tiempo real y retorna inmediatamente la categoría de estancia estimada (Corta, Media o Larga) con un tiempo de respuesta de inferencia inferior a los 200 milisegundos.

Autores
Este proyecto fue diseñado, desarrollado e implementado por:

Rafael Gutiérrez

Sara Rúa

Valentina Leal

Licencia
Este repositorio se distribuye con fines académicos. Para uso institucional o clínico del pipeline de SPARCS, por favor consulte a los autores.
