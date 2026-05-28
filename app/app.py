import streamlit as st
import joblib
import pandas as pd
import json
import sys
import traceback
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Predictor de Estadía Hospitalaria",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

MODEL_PATH = ROOT_DIR / "models" / "modelo_final_pipeline.pkl"
APR_DRG_PATH = BASE_DIR / "apr_drg_ref.json"


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
# DATOS DE REFERENCIA
# ============================================================

APR_SOI = pd.DataFrame({
    "Código": ["1", "2", "3", "4"],
    "Nivel": ["Minor", "Moderate", "Major", "Extreme"],
    "Descripción": [
        "Condición de baja complejidad, sin complicaciones significativas. Alta probabilidad de recuperación rápida.",
        "Condición de complejidad moderada. Puede requerir intervenciones adicionales o monitoreo.",
        "Condición de alta complejidad. Requiere manejo activo y puede implicar complicaciones.",
        "Condición de máxima complejidad. Alto riesgo de deterioro clínico y necesidad de cuidados intensivos."
    ]
})

APR_ROM = pd.DataFrame({
    "Nivel": ["Minor", "Moderate", "Major", "Extreme"],
    "Descripción": [
        "Riesgo de mortalidad mínimo. El diagnóstico raramente está asociado con muerte hospitalaria.",
        "Riesgo de mortalidad bajo a moderado. Requiere vigilancia clínica estándar.",
        "Riesgo de mortalidad significativo. Requiere intervención activa y monitoreo estrecho.",
        "Riesgo de mortalidad muy elevado. Alta probabilidad de desenlace fatal sin intervención intensiva."
    ]
})


@st.cache_data
def cargar_apr_drg():
    if not APR_DRG_PATH.exists():
        return pd.DataFrame(columns=["Código", "Descripción"])

    with open(APR_DRG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame([
        {"Código": str(k), "Descripción": str(v)}
        for k, v in sorted(data.items(), key=lambda x: int(x[0]))
    ])

    return df


APR_DRG_REF = cargar_apr_drg()


# ============================================================
# CARGA DEL MODELO
# ============================================================

VARIABLES_MODELO_DEFAULT = [
    "Age Group",
    "Type of Admission",
    "APR DRG Code",
    "APR MDC Code",
    "APR Severity of Illness Code",
    "APR Risk of Mortality",
    "Payment Typology 1",
    "CCS_DX_Grupo",
    "CCS_PR_Grupo"
]

OPCIONES_FALLBACK = {
    "Age Group": ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"],
    "Type of Admission": ["Elective", "Emergency", "Newborn", "Not Available", "Trauma", "Urgent"],
    "APR DRG Code": ["1"],
    "APR MDC Code": ["1"],
    "APR Severity of Illness Code": ["1", "2", "3", "4"],
    "APR Risk of Mortality": ["Minor", "Moderate", "Major", "Extreme"],
    "Payment Typology 1": ["Medicare", "Medicaid", "Private Health Insurance", "Self-Pay"],
    "CCS_DX_Grupo": ["Otros"],
    "CCS_PR_Grupo": ["Otros"]
}


def registrar_transformadores():
    """
    Registra clases personalizadas para que joblib pueda cargar el .pkl
    si fue guardado desde notebook.
    """
    try:
        from transformadores import LimpiezaInicial, ImputacionNulos, AgrupacionCCS

        sys.modules["__main__"].LimpiezaInicial = LimpiezaInicial
        sys.modules["__main__"].ImputacionNulos = ImputacionNulos
        sys.modules["__main__"].AgrupacionCCS = AgrupacionCCS

        try:
            from transformadores import CrearTarget
            sys.modules["__main__"].CrearTarget = CrearTarget
        except Exception:
            pass

    except Exception:
        pass


def buscar_onehot(pipeline):
    """
    Busca el OneHotEncoder dentro del pipeline.
    """
    if hasattr(pipeline, "named_steps"):
        for _, paso in pipeline.named_steps.items():
            if isinstance(paso, OneHotEncoder):
                return paso

    if hasattr(pipeline, "steps"):
        for _, paso in pipeline.steps:
            if isinstance(paso, OneHotEncoder):
                return paso

    return None


def crear_opciones_desde_pipeline(pipeline, variables_modelo):
    """
    Construye las opciones para los selectbox desde el OneHotEncoder entrenado.
    """
    onehot = buscar_onehot(pipeline)

    if onehot is None:
        raise ValueError("No se encontró OneHotEncoder dentro del pipeline.")

    opciones_generadas = {}

    for variable, categorias in zip(variables_modelo, onehot.categories_):
        opciones_generadas[variable] = list(categorias)

    return opciones_generadas


def ordenar_numerico_si_se_puede(lista):
    try:
        return sorted(lista, key=lambda x: int(float(x)))
    except Exception:
        return sorted(lista)


@st.cache_resource
def cargar_modelo():
    registrar_transformadores()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


modelo_ok = False
pipeline = None
artefacto = None
variables_modelo = VARIABLES_MODELO_DEFAULT.copy()
opciones = OPCIONES_FALLBACK.copy()
error_modelo = None

try:
    artefacto = cargar_modelo()

    if isinstance(artefacto, dict):
        pipeline = artefacto.get("pipeline", None)

        if pipeline is None:
            raise KeyError("El artefacto no contiene la clave 'pipeline'.")

        variables_modelo = artefacto.get("variables_modelo", VARIABLES_MODELO_DEFAULT)

        if "opciones" in artefacto and isinstance(artefacto["opciones"], dict):
            opciones = artefacto["opciones"]
        else:
            opciones = crear_opciones_desde_pipeline(pipeline, variables_modelo)

    else:
        pipeline = artefacto
        variables_modelo = VARIABLES_MODELO_DEFAULT.copy()
        opciones = crear_opciones_desde_pipeline(pipeline, variables_modelo)

    for col in VARIABLES_MODELO_DEFAULT:
        if col not in opciones or len(opciones[col]) == 0:
            opciones[col] = OPCIONES_FALLBACK[col]

    for col in ["APR DRG Code", "APR MDC Code", "APR Severity of Illness Code"]:
        if col in opciones:
            opciones[col] = ordenar_numerico_si_se_puede(opciones[col])

    for col in opciones:
        opciones[col] = [str(x) for x in opciones[col]]

    modelo_ok = True

except Exception as e:
    modelo_ok = False
    pipeline = None
    error_modelo = f"{type(e).__name__}: {repr(e)}"


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

if not modelo_ok:
    st.error(f"Error al cargar el modelo: {error_modelo}")
    with st.expander("Ver detalle técnico"):
        st.write("Ruta esperada del modelo:")
        st.code(str(MODEL_PATH))
        st.write("¿Existe el archivo?")
        st.code(str(MODEL_PATH.exists()))

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
            sorted(opciones.get("Age Group", OPCIONES_FALLBACK["Age Group"]))
        )

        tipo_admision = st.selectbox(
            "Tipo de admisión",
            sorted(opciones.get("Type of Admission", OPCIONES_FALLBACK["Type of Admission"]))
        )

        pago = st.selectbox(
            "Tipo de pago principal",
            sorted(opciones.get("Payment Typology 1", OPCIONES_FALLBACK["Payment Typology 1"]))
        )

    with col2:
        st.markdown('<div class="section-title">Clasificación Clínica</div>', unsafe_allow_html=True)

        sev_raw = st.selectbox(
            "Severidad (APR SOI)",
            ordenar_numerico_si_se_puede(
                opciones.get(
                    "APR Severity of Illness Code",
                    OPCIONES_FALLBACK["APR Severity of Illness Code"]
                )
            )
        )

        mor_raw = st.selectbox(
            "Riesgo de mortalidad (APR ROM)",
            sorted(opciones.get("APR Risk of Mortality", OPCIONES_FALLBACK["APR Risk of Mortality"]))
        )

        apr_mdc = st.selectbox(
            "Categoría diagnóstica mayor (APR MDC)",
            ordenar_numerico_si_se_puede(
                opciones.get("APR MDC Code", OPCIONES_FALLBACK["APR MDC Code"])
            )
        )

        apr_drg = st.selectbox(
            "Grupo diagnóstico APR-DRG",
            ordenar_numerico_si_se_puede(
                opciones.get("APR DRG Code", OPCIONES_FALLBACK["APR DRG Code"])
            )
        )

    with col3:
        st.markdown('<div class="section-title">Diagnóstico y Procedimiento</div>', unsafe_allow_html=True)

        dx_grupo = st.selectbox(
            "Grupo diagnóstico CCS",
            sorted(opciones.get("CCS_DX_Grupo", OPCIONES_FALLBACK["CCS_DX_Grupo"]))
        )

        pr_grupo = st.selectbox(
            "Grupo procedimiento CCS",
            sorted(opciones.get("CCS_PR_Grupo", OPCIONES_FALLBACK["CCS_PR_Grupo"]))
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Predecir Duración de Estadía")


# ============================================================
# TABS DE REFERENCIA
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
        "asignado por el sistema APR-DRG basado en diagnósticos y procedimientos."
    )
    st.dataframe(APR_SOI, hide_index=True, use_container_width=True)

with tab2:
    st.markdown(
        "**APR Risk of Mortality (ROM)** — Probabilidad estimada de muerte durante "
        "la hospitalización, asignada por el sistema APR-DRG."
    )
    st.dataframe(APR_ROM, hide_index=True, use_container_width=True)

with tab3:
    st.markdown(
        "**APR Diagnosis Related Group (DRG)** — Grupo de diagnóstico refinado que combina "
        "diagnóstico principal, procedimientos y nivel de complejidad."
    )

    if APR_DRG_REF.empty:
        st.warning("No se encontró o no se pudo cargar el archivo apr_drg_ref.json.")
    else:
        st.markdown(f"Total: **{len(APR_DRG_REF)} códigos**")

        busqueda = st.text_input(
            "Buscar por código o descripción",
            placeholder="Ej: 194 o Heart failure o Neonate",
            label_visibility="visible"
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


# ============================================================
# RESULTADO
# ============================================================

if submitted:
    if not modelo_ok:
        st.error("No se puede predecir porque el modelo no cargó correctamente.")
    else:
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

        input_data = input_data[variables_modelo]

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
                            El modelo estima una hospitalización de corta duración,
                            asociada a procedimientos rutinarios o condiciones clínicamente
                            manejables con protocolo de alta definido.
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
                            El modelo estima una hospitalización de duración moderada.
                            El paciente requiere seguimiento clínico activo y evaluación
                            periódica antes del alta.
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
                            El modelo estima una hospitalización prolongada, asociada
                            a alta complejidad clínica, riesgo de complicaciones o
                            necesidad de intervenciones múltiples.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_info:
                st.markdown('<div class="section-title">Resumen de entrada</div>', unsafe_allow_html=True)
                st.dataframe(
                    pd.DataFrame({
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
                    }),
                    hide_index=True,
                    use_container_width=True,
                    height=340
                )

            st.markdown("""
            <div class="disclaimer">
                <b>Aviso importante sobre el uso de esta herramienta</b><br><br>
                Este sistema utiliza un modelo de clasificación supervisada
                (<b>Support Vector Machine</b>) entrenado con datos históricos
                del sistema SPARCS del Estado de Nueva York (2015), con un
                <b>F1-macro de 0.59</b> sobre datos de prueba independientes.<br><br>
                El <b>F1-macro</b> mide el balance entre precisión y recall
                promediado entre las tres categorías de estadía (Corta, Media, Larga),
                otorgando igual peso a cada clase. Un valor de 0.59 indica capacidad
                de clasificación moderada.<br><br>
                <b>Esta predicción es únicamente una herramienta de apoyo a la
                planificación administrativa y NO reemplaza el criterio clínico del
                personal médico.</b> Las decisiones sobre hospitalización, tratamiento
                y alta deben tomarse exclusivamente por profesionales de la salud habilitados.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error en la predicción: {type(e).__name__}: {repr(e)}")
            with st.expander("Ver detalle técnico"):
                st.code(traceback.format_exc())
