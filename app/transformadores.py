import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LimpiezaInicial(BaseEstimator, TransformerMixin):
    ELIMINAR = [
        "Operating Certificate Number", "Facility Id", "Facility Name",
        "Zip Code - 3 digits", "Discharge Year", "Abortion Edit Indicator",
        "Attending Provider License Number", "Operating Provider License Number",
        "Other Provider License Number", "CCS Diagnosis Description",
        "CCS Procedure Description", "APR DRG Description", "APR MDC Description",
        "APR Severity of Illness Description", "Payment Typology 2",
        "Payment Typology 3", "Patient Disposition", "Total Charges", "Total Costs",
        "Ethnicity", "Gender", "APR Medical Surgical Description",
        "Hospital County", "Race", "Birth Weight", "Emergency Department Indicator",
        "Health Service Area"
    ]
    CATEGORICAS = [
        "Age Group", "Type of Admission", "APR Risk of Mortality",
        "Payment Typology 1", "APR MDC Code", "APR Severity of Illness Code",
        "APR DRG Code", "CCS Diagnosis Code", "CCS Procedure Code"
    ]
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X.copy()
        df = df.drop(columns=[c for c in self.ELIMINAR if c in df.columns])
        for col in self.CATEGORICAS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

class ImputacionNulos(BaseEstimator, TransformerMixin):
    IMPUTAR_MODA = ["APR Risk of Mortality"]
    def fit(self, X, y=None):
        self.modas_ = {}
        for col in self.IMPUTAR_MODA:
            if col in X.columns:
                self.modas_[col] = X[col].mode()[0]
        return self
    def transform(self, X):
        df = X.copy()
        for col, moda in self.modas_.items():
            if col in df.columns:
                df[col] = df[col].fillna(moda)
                if hasattr(df[col], "cat"):
                    df[col] = df[col].cat.remove_unused_categories()
        return df

class AgrupacionCCS(BaseEstimator, TransformerMixin):
    DX_GRUPOS = {
        **{c: "Infecciosas"            for c in range(1, 11)},
        **{c: "Cancer"                 for c in range(11, 46)},
        **{c: "Neoplasias_benignas"    for c in range(46, 48)},
        **{c: "Endocrinas_metabolicas" for c in range(48, 65)},
        **{c: "Mentales"               for c in list(range(650, 664)) + [670]},
        **{c: "Nervioso_sensorial"     for c in range(76, 96)},
        **{c: "Circulatorio"           for c in range(96, 122)},
        **{c: "Respiratorio"           for c in range(122, 135)},
        **{c: "Digestivo"              for c in range(135, 156)},
        **{c: "Genitourinario"         for c in range(156, 176)},
        **{c: "Embarazo_parto"         for c in range(176, 197)},
        **{c: "Piel"                   for c in range(197, 201)},
        **{c: "Musculoesqueletico"     for c in range(201, 213)},
        **{c: "Congenitas"             for c in range(213, 218)},
        **{c: "Perinatales"            for c in range(218, 225)},
        **{c: "Sintomas_mal_definidos" for c in range(225, 245)},
        **{c: "Lesiones"               for c in range(245, 260)},
        917: "Especiales"
    }
    PR_GRUPOS = {
        0: "Sin_procedimiento",
        **{c: "Sistema_nervioso"        for c in range(1, 10)},
        **{c: "Sistema_endocrino"       for c in range(10, 13)},
        **{c: "Ojos_oidos"              for c in range(13, 22)},
        **{c: "Cardiovascular"          for c in range(22, 48)},
        **{c: "Respiratorio"            for c in range(48, 56)},
        **{c: "Digestivo"               for c in range(56, 84)},
        **{c: "Urinario_renal"          for c in range(84, 101)},
        **{c: "Ginecologia_obstetricia" for c in range(101, 140)},
        **{c: "Musculoesqueletico"      for c in range(140, 168)},
        **{c: "Tegumentario"            for c in range(168, 176)},
        **{c: "Hematologia_inmunologia" for c in range(176, 182)},
        **{c: "Diagnostico"             for c in range(182, 210)},
        **{c: "Terapeutico_miscelaneo"  for c in range(210, 232)},
        999: "Otros"
    }
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X.copy()
        if "CCS Diagnosis Code" in df.columns:
            df["CCS_DX_Grupo"] = df["CCS Diagnosis Code"].astype(int).map(
                self.DX_GRUPOS).fillna("Otros").astype("category")
            df = df.drop(columns=["CCS Diagnosis Code"])
        if "CCS Procedure Code" in df.columns:
            df["CCS_PR_Grupo"] = df["CCS Procedure Code"].astype(int).map(
                self.PR_GRUPOS).fillna("Otros").astype("category")
            df = df.drop(columns=["CCS Procedure Code"])
        return df
