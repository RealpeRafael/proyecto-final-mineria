import streamlit as st
import joblib
import pandas as pd
import json
from pathlib import Path

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Predictor de Estadía Hospitalaria",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# Posibles rutas del modelo
POSIBLES_MODELOS = [
    BASE_DIR / "models" / "modelo_final_pipeline.pkl",
    BASE_DIR / "modelo_final_pipeline.pkl",
    Path("models/modelo_final_pipeline.pkl"),
    Path("modelo_final_pipeline.pkl")
]

POSIBLES_JSON = [
    BASE_DIR / "apr_drg_ref.json",
    Path("apr_drg_ref.json")
]

# ============================================================
# ESTILOS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #F8FAFC !important;
}

[data-testid="stSidebar"] {
    background-color: #0D1B2A !important;
    border-right: 1px solid #1A2B3C;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

.section-title {
    font-size: 10px;
    font-weight: 600;
    color: #1E3A8A;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #E2E8F0;
}

.stSelectbox label {
    font-size: 11px !important;
    font-weight: 500 !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

.stFormSubmitButton > button {
    background-color: #0D1B2A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 40px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
}

.stFormSubmitButton > button:hover {
    background-color: #1E3A8A !important;
}

.result-card {
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 16px;
}

.result-corta {
    background: #F0FDFA;
    border: 1px solid #0EA5A4;
    border-left: 5px solid #0EA5A4;
}

.result-media {
    background: #FEFCE8;
    border: 1px solid #EAB308;
    border-left: 5px solid #EAB308;
}

.result-larga {
    background: #FFF1F2;
    border: 1px solid #F43F5E;
    border-left: 5px solid #F43F5E;
}

.result-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}

.label-corta { color: #0EA5A4; }
.label-media { color: #CA8A04; }
.label-larga { color: #E11D48; }

.result-title {
    font-size: 26px;
    font-weight: 700;
    color: #0D1B2A;
    margin-bottom: 2px;
    line-height: 1.1;
}

.result-days {
    font-size: 13px;
    color: #64748B;
    margin-bottom: 12px;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
}

.result-desc {
    font-size: 13px;
    color: #475569;
    line-height: 1.65;
}

.disclaimer {
    background: #F8FAFC;
    border-left: 3px solid #1E3A8A;
    border-radius: 6px;
    padding: 16px 20px;
    font-size: 11.5px;
    color: #64748B;
    line-height: 1.75;
    margin-top: 24px;
}

.disclaimer b {
    color: #0D1B2A;
}

.ref-item {
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 12px;
    line-height: 1.5;
    border-left: 3px solid;
}

.ref-corta {
    background: rgba(14,165,164,0.1);
    border-color: #0EA5A4;
}

.ref-media {
    background: rgba(234,179,8,0.1);
    border-color: #EAB308;
}

.ref-larga {
    background: rgba(244,63,94,0.1);
    border-color: #F43F5E;
}

.ref-item b {
    color: #F1F5F9 !important;
    font-size: 12px;
}

.ref-item small {
    color: #94A3B8 !important;
    font-size: 11px;
}

hr {
    border-color: #E2E8F0 !important;
    margin: 20px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CLASES NECESARIAS PARA QUE EL PKL CARGUE
# IMPORTANTE: no borrar esto
# ============================================================

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class LimpiezaInicial(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy()


class ImputacionNulos(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy()


class AgrupacionCCS(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy()


class CrearTarget(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.copy()


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def encontrar_archivo(posibles_rutas):
    for ruta in posibles_rutas:
        if ruta.exists():
            return ruta
    return None


def ordenar_valores(lista):
    lista = list(lista)
    try:
        return sorted(lista, key=lambda x: int(x))
    except Exception:
        return sorted(lista, key=lambda x: str(x))


@st.cache_resource
def cargar_modelo():
    ruta_modelo = encontrar_archivo(POSIBLES_MODELOS)

    if ruta_modelo is None:
        raise FileNotFoundError(
            "No se encontró modelo_final_pipeline.pkl. "
            "Verifica que esté en app/models/ o en la misma carpeta de app.py."
        )

    artefacto = joblib.load(ruta_modelo)
    return artefacto


def construir_opciones_desde_pipeline(artefacto):
    """
    El .pkl no trae 'opciones'. Entonces se construyen a partir
    del OneHotEncoder ya entrenado dentro del pipeline.
    """

    pipeline = artefacto["pipeline"]
    variables_modelo = artefacto["variables_modelo"]

    onehot = pipeline.named_steps["onehot"]

    opciones = {}

    for variable, categorias in zip(variables_modelo, onehot.categories_):
        opciones[variable] = ordenar_valores([str(c) for c in categorias])

    return opciones


@st.cache_data
def cargar_apr_drg():
    ruta_json = encontrar_archivo(POSIBLES_JSON)

    if ruta_json is None:
        return pd.DataFrame({
            "Código": [],
            "Descripción": []
        })

    with open(ruta_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame([
        {"Código": str(k), "Descripción": str(v)}
        for k, v in data.items()
    ])

    try:
        df = df.sort_values("Código", key=lambda s: s.astype(int))
    except Exception:
        df = df.sort_values("Código")

    return df.reset_index(drop=True)


# ============================================================
# DATOS DE REFERENCIA
# ============================================================

APR_SOI = pd.DataFrame({
    "Código": ["1", "2", "3", "4"],
    "Nivel": ["Minor", "Moderate", "Major", "Extreme"],
    "Descripción": [
        "Condición de baja complejidad, sin complicaciones significativas.",
        "Condición de complejidad moderada. Puede requerir monitoreo adicional.",
        "Condición de alta complejidad. Requiere manejo clínico activo.",
        "Condición de máxima complejidad. Alto riesgo de deterioro clínico."
    ]
})

APR_ROM = pd.DataFrame({
    "Nivel": ["Minor", "Moderate", "Major", "Extreme"],
    "Descripción": [
        "Riesgo de mortalidad mínimo.",
        "Riesgo de mortalidad bajo a moderado.",
        "Riesgo de mortalidad significativo.",
        "Riesgo de mortalidad muy elevado."
    ]
})

APR_DRG_REF = cargar_apr_drg()

# ============================================================
# CARGA DEL MODELO
# ============================================================

modelo_ok = False
artefacto = None
pipeline = None
opciones = {}

try:
    artefacto = cargar_modelo()
    pipeline = artefacto["pipeline"]
    opciones = construir_opciones_desde_pipeline(artefacto)
    modelo_ok = True

except Exception as e:
    st.error(f"Error al cargar el modelo: {type(e).__name__}: {e}")
    modelo_ok = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:28px 20px 20px;">
        <div style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px;">
            Sistema de Apoyo Clínico
        </div>
        <div style="font-size:16px;font-weight:700;color:#FFFFFF;line-height:1.3;">
            Predicción de Estadía Hospitalaria
        </div>
        <div style="margin-top:16px;border-top:1px solid #1E3A8A;"></div>
    </div>

    <div style="padding:0 20px 20px;">
        <div style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px;">
            Instrucciones
        </div>
        <div style="font-size:12px;color:#94A3B8;line-height:2.1;">
            1. Complete todos los campos<br>
            2. Presione <span style="color:#0EA5A4;font-weight:600;">Predecir</span><br>
            3. Revise el resultado y el aviso legal
        </div>
        <div style="margin-top:16px;border-top:1px solid #1E3A8A;"></div>
    </div>

    <div style="padding:0 20px 20px;">
        <div style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:12px;">
            Categorías de estadía
        </div>
        <div class="ref-item ref-corta">
            <b>Estadía Corta — 1 a 3 días</b><br>
            <small>Procedimientos rutinarios o condiciones manejables</small>
        </div>
        <div class="ref-item ref-media">
            <b>Estadía Media — 4 a 7 días</b><br>
            <small>Tratamientos con seguimiento clínico activo</small>
        </div>
        <div class="ref-item ref-larga">
            <b>Estadía Larga — 8 días o más</b><br>
            <small>Casos complejos o con riesgo de complicaciones</small>
        </div>
        <div style="margin-top:16px;border-top:1px solid #1E3A8A;"></div>
    </div>

    <div style="padding:0 20px;">
        <div style="font-size:10px;color:#334155;">
            v1.0.0 &nbsp;·&nbsp; SVM &nbsp;·&nbsp; SPARCS NY 2015
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style="margin-bottom:28px;">
    <div style="font-size:10px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px;">
        Minería de Datos · Universidad Pontificia Bolivariana
    </div>
    <div style="font-size:24px;font-weight:700;color:#0D1B2A;line-height:1.2;">
        Predictor de Duración de Estadía Hospitalaria
    </div>
    <div style="font-size:13px;color:#64748B;margin-top:6px;">
        Herramienta de apoyo a la planificación de recursos · SPARCS New York State 2015
    </div>
    <div style="margin-top:16px;border-bottom:1px solid #E2E8F0;"></div>
</div>
""", unsafe_allow_html=True)


if not modelo_ok:
    st.stop()

# ============================================================
# FORMULARIO
# ============================================================

st.markdown('<div class="section-title">Datos del Paciente</div>', unsafe_allow_html=True)

with st.form("formulario_prediccion"):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="section-title">Información General</div>', unsafe_allow_html=True)

        age_group = st.selectbox(
            "Grupo de edad",
            opciones["Age Group"]
        )

        tipo_admision = st.selectbox(
            "Tipo de admisión",
            opciones["Type of Admission"]
        )

        pago = st.selectbox(
            "Tipo de pago principal",
            opciones["Payment Typology 1"]
        )

    with col2:
        st.markdown('<div class="section-title">Clasificación Clínica</div>', unsafe_allow_html=True)

        sev_raw = st.selectbox(
            "Severidad (APR SOI)",
            opciones["APR Severity of Illness Code"]
        )

        mor_raw = st.selectbox(
            "Riesgo de mortalidad (APR ROM)",
            opciones["APR Risk of Mortality"]
        )

        apr_mdc = st.selectbox(
            "Categoría diagnóstica mayor (APR MDC)",
            opciones["APR MDC Code"]
        )

        apr_drg = st.selectbox(
            "Grupo diagnóstico APR-DRG",
            opciones["APR DRG Code"]
        )

    with col3:
        st.markdown('<div class="section-title">Diagnóstico y Procedimiento</div>', unsafe_allow_html=True)

        dx_grupo = st.selectbox(
            "Grupo diagnóstico CCS",
            opciones["CCS_DX_Grupo"]
        )

        pr_grupo = st.selectbox(
            "Grupo procedimiento CCS",
            opciones["CCS_PR_Grupo"]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button("Predecir Duración de Estadía")


# ============================================================
# REFERENCIAS
# ============================================================

st.markdown(
    '<div class="section-title" style="margin-top:28px;">Referencia de Códigos Clínicos</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "APR Severidad (SOI)",
    "APR Riesgo de Mortalidad (ROM)",
    "APR Grupo Diagnóstico (DRG)"
])

with tab1:
    st.markdown(
        "**APR Severity of Illness (SOI)** — Nivel de complejidad clínica del paciente "
        "asignado por el sistema APR-DRG."
    )
    st.dataframe(APR_SOI, hide_index=True, use_container_width=True)

with tab2:
    st.markdown(
        "**APR Risk of Mortality (ROM)** — Probabilidad estimada de muerte durante "
        "la hospitalización."
    )
    st.dataframe(APR_ROM, hide_index=True, use_container_width=True)

with tab3:
    st.markdown(
        "**APR Diagnosis Related Group (DRG)** — Grupo de diagnóstico refinado que combina "
        "diagnóstico principal, procedimientos y nivel de complejidad."
    )

    if len(APR_DRG_REF) > 0:
        busqueda = st.text_input(
            "Buscar por código o descripción",
            placeholder="Ej: 194, Heart failure, Neonate"
        )

        if busqueda.strip():
            filtro = APR_DRG_REF[
                APR_DRG_REF["Código"].str.contains(busqueda.strip(), case=False, na=False) |
                APR_DRG_REF["Descripción"].str.contains(busqueda.strip(), case=False, na=False)
            ]

            st.caption(f"{len(filtro)} resultado(s) encontrado(s)")
            st.dataframe(filtro, hide_index=True, use_container_width=True, height=400)

        else:
            st.dataframe(APR_DRG_REF, hide_index=True, use_container_width=True, height=400)

    else:
        st.warning("No se encontró el archivo apr_drg_ref.json.")


# ============================================================
# PREDICCIÓN
# ============================================================

if submitted:
    input_data = pd.DataFrame([{
        "Age Group": age_group,
        "Type of Admission": tipo_admision,
        "APR DRG Code": apr_drg,
        "APR MDC Code": apr_mdc,
        "APR Severity of Illness Code": sev_raw,
        "APR Risk of Mortality": mor_raw,
        "Payment Typology 1": pago,
        "CCS_DX_Grupo": dx_grupo,
        "CCS_PR_Grupo": pr_grupo
    }])

    # Asegurar mismo orden de columnas que el entrenamiento
    input_data = input_data[artefacto["variables_modelo"]]

    try:
        prediccion = pipeline.predict(input_data)[0]

        st.markdown(
            '<div class="section-title" style="margin-top:24px;">Resultado</div>',
            unsafe_allow_html=True
        )

        col_res, col_info = st.columns([1.2, 0.8], gap="large")

        with col_res:
            if prediccion == "Corta":
                st.markdown("""
                <div class="result-card result-corta">
                    <div class="result-label label-corta">Resultado</div>
                    <div class="result-title">Estadía Corta</div>
                    <div class="result-days">1 a 3 días estimados</div>
                    <div class="result-desc">
                        El modelo estima una hospitalización de corta duración, asociada a procedimientos rutinarios
                        o condiciones clínicamente manejables.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif prediccion == "Media":
                st.markdown("""
                <div class="result-card result-media">
                    <div class="result-label label-media">Resultado</div>
                    <div class="result-title">Estadía Media</div>
                    <div class="result-days">4 a 7 días estimados</div>
                    <div class="result-desc">
                        El modelo estima una hospitalización de duración moderada. El paciente podría requerir
                        seguimiento clínico activo antes del alta.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="result-card result-larga">
                    <div class="result-label label-larga">Resultado</div>
                    <div class="result-title">Estadía Larga</div>
                    <div class="result-days">8 días o más estimados</div>
                    <div class="result-desc">
                        El modelo estima una hospitalización prolongada, asociada a mayor complejidad clínica,
                        riesgo de complicaciones o necesidad de múltiples intervenciones.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_info:
            st.markdown('<div class="section-title">Resumen de entrada</div>', unsafe_allow_html=True)

            resumen = pd.DataFrame({
                "Variable": [
                    "Edad",
                    "Admisión",
                    "Severidad",
                    "Mortalidad",
                    "Diagnóstico CCS",
                    "Procedimiento CCS",
                    "APR MDC",
                    "APR DRG",
                    "Pago"
                ],
                "Valor": [
                    age_group,
                    tipo_admision,
                    sev_raw,
                    mor_raw,
                    dx_grupo,
                    pr_grupo,
                    apr_mdc,
                    apr_drg,
                    pago
                ]
            })

            st.dataframe(
                resumen,
                hide_index=True,
                use_container_width=True,
                height=340
            )

        st.markdown("""
        <div class="disclaimer">
            <b>Aviso importante sobre el uso de esta herramienta</b><br><br>
            Este sistema utiliza un modelo de clasificación supervisada entrenado con datos históricos
            del sistema SPARCS del Estado de Nueva York 2015.<br><br>
            Esta predicción es únicamente una herramienta de apoyo a la planificación administrativa
            y <b>no reemplaza el criterio clínico del personal médico</b>.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error en la predicción: {type(e).__name__}: {e}")
