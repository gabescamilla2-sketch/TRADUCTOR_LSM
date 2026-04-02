"""


El script:
  1. Limpia y normaliza el CSV hacia 63 features unificados
  2. Agrega features de las secuencias (J, K, Z) con descriptores estadísticos
  3. Entrena tres modelos: Random Forest, SVM y MLP
  4. Evalúa y guarda el mejor modelo + artefactos de inferencia

Uso:
    python entrenar_modelo.py

Salidas (en data/processed/):
    mejor_modelo.pkl      -> modelo sklearn listo para inferencia
    label_encoder.pkl     -> LabelEncoder para decodificar predicciones
    scaler.pkl            -> StandardScaler (fit sobre train)
    info_modelo.json      -> métricas, clases, fecha de entrenamiento
    confusion_matrix.png  -> matriz de confusión del mejor modelo
    curvas_aprendizaje.png -> overfitting check
"""

import json
import warnings
import joblib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─── Configuración ─────────────────────────────────────────────────────────────

ARCHIVO_CSV        = Path("data/processed/dataset_lsm.csv")
ARCHIVO_SECUENCIAS = Path("data/processed/secuencias_lsm.json")
DIR_SALIDA         = Path("data/processed")
SENAS_MOVIMIENTO   = {"J", "K", "Z"}
RANDOM_STATE       = 42
TEST_SIZE          = 0.20   # 20% test
VAL_SIZE           = 0.10   # 10% validación (del total)
N_FRAMES_FIJO      = 26     # frames esperados por secuencia

# ─── 1. Carga y limpieza del CSV ───────────────────────────────────────────────

def cargar_y_limpiar_csv() -> pd.DataFrame:
    """
    Carga el CSV y lo normaliza a 63 features unificados.

    El CSV actual tiene:
      - Cols 1-63  (x0..z20):           mano 1 (sin prefijo en header)
      - Cols 64-126 (Unnamed: 64..126): mano 2 (header faltó en versión antigua)

    En cada fila, solo UNA de las dos mitades tiene valores reales;
    la otra es todo ceros. Se colapsa a un único vector de 63 features
    tomando la mitad activa.
    """
    print("=" * 55)
    print("1. CARGA Y LIMPIEZA DEL DATASET")
    print("=" * 55)

    df = pd.read_csv(ARCHIVO_CSV)

    # Eliminar fila con etiqueta nula (encabezado duplicado o registro vacío)
    antes = len(df)
    df = df.dropna(subset=["etiqueta"]).reset_index(drop=True)
    print(f"   Filas eliminadas (etiqueta nula): {antes - len(df)}")
    print(f"   Muestras válidas: {len(df)}")

    # Separar las dos mitades
    feature_cols = [c for c in df.columns if c != "etiqueta"]
    primera_mitad = feature_cols[:63]    # mano capturada en primer slot
    segunda_mitad = feature_cols[63:126] # mano capturada en segundo slot

    # Identificar qué slot tiene la mano activa por fila
    primera_activa = (df[primera_mitad] != 0).any(axis=1)
    segunda_activa = (df[segunda_mitad] != 0).any(axis=1)
    ambas_activas  = primera_activa & segunda_activa

    print(f"\n   Distribución de manos en el dataset:")
    print(f"     Solo primera mitad activa (mano izq): {(primera_activa & ~segunda_activa).sum()}")
    print(f"     Solo segunda mitad activa (mano der): {(~primera_activa & segunda_activa).sum()}")
    print(f"     Ambas activas simultáneamente:        {ambas_activas.sum()}")

    # Construir array unificado de 63 features
    # Prioridad: primera mitad si está activa, si no usar segunda
    X_primera = df[primera_mitad].values.astype(np.float32)
    X_segunda  = df[segunda_mitad].values.astype(np.float32)

    # Para filas donde ambas están activas (raro), promediar
    X_unificado = np.where(
        primera_activa.values[:, None],
        X_primera,
        X_segunda
    )
    # Casos donde ambas están activas: promedio
    if ambas_activas.any():
        mask = ambas_activas.values
        X_unificado[mask] = (X_primera[mask] + X_segunda[mask]) / 2.0

    # Nombres de columna limpios para el resultado
    nombres_cols = []
    for i in range(21):
        nombres_cols += [f"x{i}", f"y{i}", f"z{i}"]

    df_limpio = pd.DataFrame(X_unificado, columns=nombres_cols)
    df_limpio.insert(0, "etiqueta", df["etiqueta"].values)

    print(f"\n   Dataset unificado: {df_limpio.shape[0]} filas × {df_limpio.shape[1]-1} features")
    print(f"   Señas: {sorted(df_limpio['etiqueta'].unique())}")
    print(f"   Muestras por seña:\n   {df_limpio['etiqueta'].value_counts().sort_index().to_dict()}")

    return df_limpio


# ─── 2. Features de secuencias ─────────────────────────────────────────────────

def extraer_features_secuencia(frames: list) -> np.ndarray:
    """
    Convierte una lista de frames (cada frame = 126 floats) en un
    vector de features estadísticos de longitud fija.

    Para cada uno de los 63 features activos extrae:
      media, std, min, max, rango, pendiente_lineal
    → 63 × 6 = 378 features por secuencia.

    Se usa la mitad activa igual que en el CSV estático.
    """
    arr = np.array(frames, dtype=np.float32)  # shape: (n_frames, 126)

    # Determinar qué mitad está activa (misma lógica que en CSV)
    primera = arr[:, :63]
    segunda = arr[:, 63:126]

    primera_activa = (primera != 0).any()
    datos = primera if primera_activa else segunda

    # Padding/truncado a N_FRAMES_FIJO para consistencia
    n = datos.shape[0]
    if n < N_FRAMES_FIJO:
        pad = np.zeros((N_FRAMES_FIJO - n, 63), dtype=np.float32)
        datos = np.vstack([datos, pad])
    else:
        datos = datos[:N_FRAMES_FIJO]

    # Estadísticos por feature a lo largo del tiempo
    media     = datos.mean(axis=0)
    std       = datos.std(axis=0)
    minimo    = datos.min(axis=0)
    maximo    = datos.max(axis=0)
    rango     = maximo - minimo

    # Pendiente lineal (velocidad promedio del movimiento)
    t = np.arange(N_FRAMES_FIJO)
    pendiente = np.polyfit(t, datos, 1)[0]  # coef. lineal por feature

    return np.concatenate([media, std, minimo, maximo, rango, pendiente])


def cargar_secuencias() -> pd.DataFrame | None:
    """Carga el JSON de secuencias y extrae features estadísticos."""
    if not ARCHIVO_SECUENCIAS.exists():
        print("   [AVISO] No se encontró archivo de secuencias.")
        return None

    print("\n" + "=" * 55)
    print("2. CARGA DE SECUENCIAS (J, K, Z)")
    print("=" * 55)

    with open(ARCHIVO_SECUENCIAS, "r", encoding="utf-8") as f:
        data = json.load(f)

    registros = []
    for sena, secuencias in data.items():
        print(f"   {sena}: {len(secuencias)} secuencias")
        for seq in secuencias:
            frames = seq.get("frames", [])
            if len(frames) < 5:  # descartar secuencias muy cortas
                continue
            feats = extraer_features_secuencia(frames)
            registros.append([sena] + feats.tolist())

    if not registros:
        print("   [AVISO] Sin secuencias válidas.")
        return None

    n_feats = len(registros[0]) - 1
    col_names = [f"sf_{i}" for i in range(n_feats)]
    df_seq = pd.DataFrame(registros, columns=["etiqueta"] + col_names)

    print(f"\n   Features extraídos por secuencia: {n_feats}")
    print(f"   Total registros de secuencia: {len(df_seq)}")
    return df_seq


# ─── 3. Preparación de datos para entrenamiento ────────────────────────────────

def preparar_datos(df_estatico: pd.DataFrame, df_secuencias: pd.DataFrame | None):
    """
    Combina datos estáticos y de secuencia en un dataset listo para entrenar.
    Si hay secuencias, los modelos se entrenan por separado o con padding.

    Retorna: X_train, X_val, X_test, y_train, y_val, y_test, label_encoder
    """
    print("\n" + "=" * 55)
    print("3. PREPARACIÓN PARA ENTRENAMIENTO")
    print("=" * 55)

    # ── Dataset estático (señas sin movimiento) ──
    mask_estaticas = ~df_estatico["etiqueta"].isin(SENAS_MOVIMIENTO)
    df_train_est = df_estatico[mask_estaticas].copy()

    X_est = df_train_est.drop("etiqueta", axis=1).values.astype(np.float32)
    y_est = df_train_est["etiqueta"].values

    print(f"\n   Muestras estáticas para entrenar: {len(X_est)}")
    print(f"   Features estáticos: {X_est.shape[1]}")

    # ── Dataset de secuencias ──
    X_seq, y_seq = None, None
    n_feats_seq = 0

    if df_secuencias is not None and len(df_secuencias) > 0:
        X_seq = df_secuencias.drop("etiqueta", axis=1).values.astype(np.float32)
        y_seq = df_secuencias["etiqueta"].values
        n_feats_seq = X_seq.shape[1]
        print(f"   Muestras de secuencia: {len(X_seq)}")
        print(f"   Features de secuencia: {n_feats_seq}")

        # Padding de datos estáticos para igualar dimensiones
        pad_est = np.zeros((X_est.shape[0], n_feats_seq), dtype=np.float32)
        X_est_pad = np.hstack([X_est, pad_est])

        # Padding de secuencias en la parte estática (ceros)
        pad_seq = np.zeros((X_seq.shape[0], X_est.shape[1]), dtype=np.float32)
        X_seq_pad = np.hstack([pad_seq, X_seq])

        X_total = np.vstack([X_est_pad, X_seq_pad])
        y_total = np.concatenate([y_est, y_seq])
        print(f"\n   ⚠️  Secuencias insuficientes (<50). Se entrenan combinados.")
        print(f"   Features totales (estático+secuencia padded): {X_total.shape[1]}")
    else:
        X_total = X_est
        y_total = y_est

    # ── Codificar etiquetas ──
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_total)

    print(f"\n   Clases a clasificar ({len(le.classes_)}): {list(le.classes_)}")

    # ── Split estratificado: train / val / test ──
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_total, y_encoded,
        test_size=TEST_SIZE,
        stratify=y_encoded,
        random_state=RANDOM_STATE
    )
    val_adj = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_adj,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    print(f"\n   División del dataset:")
    print(f"     Train:      {len(X_train):>5} muestras ({100*(1-TEST_SIZE-VAL_SIZE):.0f}%)")
    print(f"     Validación: {len(X_val):>5} muestras ({100*VAL_SIZE:.0f}%)")
    print(f"     Test:       {len(X_test):>5} muestras ({100*TEST_SIZE:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, le


# ─── 4. Entrenamiento de modelos ───────────────────────────────────────────────

def construir_modelos() -> dict:
    """Define los tres modelos candidatos con sus hiperparámetros."""
    return {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,   # necesario para predict_proba en inferencia
                random_state=RANDOM_STATE
            ))
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=RANDOM_STATE
            ))
        ]),
    }


def entrenar_y_evaluar(
    modelos: dict,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    le: LabelEncoder
) -> tuple:
    """
    Entrena cada modelo, evalúa en validación con CV,
    y retorna (mejor_nombre, mejor_modelo, resultados_dict).
    """
    print("\n" + "=" * 55)
    print("4. ENTRENAMIENTO Y EVALUACIÓN")
    print("=" * 55)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    resultados = {}

    for nombre, pipeline in modelos.items():
        print(f"\n   [{nombre}]")

        # Cross-validation en train
        scores_cv = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="f1_weighted", n_jobs=-1
        )
        print(f"     CV F1 (5-fold): {scores_cv.mean():.4f} ± {scores_cv.std():.4f}")

        # Entrenar con todo el train set
        pipeline.fit(X_train, y_train)

        # Evaluación en validación
        y_pred_val = pipeline.predict(X_val)
        acc_val  = accuracy_score(y_val, y_pred_val)
        f1_val   = f1_score(y_val, y_pred_val, average="weighted", zero_division=0)
        print(f"     Accuracy  val: {acc_val:.4f}")
        print(f"     F1 weighted val: {f1_val:.4f}")

        resultados[nombre] = {
            "pipeline":  pipeline,
            "cv_f1_mean": float(scores_cv.mean()),
            "cv_f1_std":  float(scores_cv.std()),
            "val_accuracy": float(acc_val),
            "val_f1":    float(f1_val),
        }

    # Seleccionar mejor modelo por F1 en validación
    mejor_nombre = max(resultados, key=lambda k: resultados[k]["val_f1"])
    mejor_pipeline = resultados[mejor_nombre]["pipeline"]

    print(f"\n   🏆  Mejor modelo: {mejor_nombre} "
          f"(val F1={resultados[mejor_nombre]['val_f1']:.4f})")

    # Evaluación final en test (solo una vez)
    print("\n" + "=" * 55)
    print("5. EVALUACIÓN FINAL EN TEST SET")
    print("=" * 55)

    y_pred_test = mejor_pipeline.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test  = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)

    print(f"\n   Accuracy  test: {acc_test:.4f}")
    print(f"   F1 weighted test: {f1_test:.4f}")
    print(f"\n   Reporte completo:\n")
    print(classification_report(
        y_test, y_pred_test,
        target_names=le.classes_,
        zero_division=0
    ))

    # Guardar métricas
    for nombre in resultados:
        resultados[nombre].pop("pipeline")  # no serializable en JSON

    return mejor_nombre, mejor_pipeline, resultados, acc_test, f1_test, y_pred_test


# ─── 5. Visualizaciones ────────────────────────────────────────────────────────

def graficar_confusion(y_test, y_pred, le: LabelEncoder, nombre_modelo: str):
    """Genera y guarda la matriz de confusión normalizada."""
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    clases = le.classes_

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt=".2f",
        xticklabels=clases, yticklabels=clases,
        cmap="Blues", linewidths=0.5, ax=ax
    )
    ax.set_title(f"Matriz de Confusión — {nombre_modelo}\n(normalizada por fila)", fontsize=13)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    plt.tight_layout()

    ruta = DIR_SALIDA / "confusion_matrix.png"
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"\n   Matriz guardada: {ruta}")


def graficar_comparacion_modelos(resultados: dict):
    """Barra comparativa de F1 por modelo."""
    nombres = list(resultados.keys())
    f1_vals = [resultados[n]["val_f1"] for n in nombres]
    cv_vals = [resultados[n]["cv_f1_mean"] for n in nombres]

    x = np.arange(len(nombres))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, cv_vals, width, label="CV F1 (train)", color="#4C72B0")
    bars2 = ax.bar(x + width/2, f1_vals, width, label="F1 Validación", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score (weighted)")
    ax.set_title("Comparación de modelos")
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9)

    plt.tight_layout()
    ruta = DIR_SALIDA / "comparacion_modelos.png"
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"   Comparación guardada: {ruta}")


# ─── 6. Guardado de artefactos ─────────────────────────────────────────────────

def guardar_artefactos(
    pipeline,
    le: LabelEncoder,
    nombre_modelo: str,
    resultados: dict,
    acc_test: float,
    f1_test: float
):
    """Guarda el modelo y todos los artefactos necesarios para inferencia."""
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    # Modelo completo (pipeline con scaler incluido)
    ruta_modelo = DIR_SALIDA / "mejor_modelo.pkl"
    joblib.dump(pipeline, ruta_modelo)

    # LabelEncoder
    ruta_le = DIR_SALIDA / "label_encoder.pkl"
    joblib.dump(le, ruta_le)

    # Scaler por separado (útil para inferencia en tiempo real)
    ruta_scaler = DIR_SALIDA / "scaler.pkl"
    joblib.dump(pipeline.named_steps["scaler"], ruta_scaler)

    # Info del modelo (para cargar correctamente en inferencia)
    info = {
        "modelo": nombre_modelo,
        "clases": list(le.classes_),
        "num_clases": len(le.classes_),
        "test_accuracy": round(acc_test, 4),
        "test_f1_weighted": round(f1_test, 4),
        "resultados_modelos": resultados,
        "features": {
            "total": pipeline.named_steps["clf"].n_features_in_
                     if hasattr(pipeline.named_steps["clf"], "n_features_in_")
                     else "N/A",
            "descripcion": "63 landmarks (x,y,z × 21 puntos) de la mano activa"
        },
        "fecha_entrenamiento": datetime.now().isoformat(),
        "archivos": {
            "modelo":        str(ruta_modelo),
            "label_encoder": str(ruta_le),
            "scaler":        str(ruta_scaler),
        }
    }

    ruta_info = DIR_SALIDA / "info_modelo.json"
    with open(ruta_info, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("6. ARTEFACTOS GUARDADOS")
    print("=" * 55)
    print(f"   Modelo:        {ruta_modelo}")
    print(f"   LabelEncoder:  {ruta_le}")
    print(f"   Scaler:        {ruta_scaler}")
    print(f"   Info JSON:     {ruta_info}")


# ─── 7. Snippet de inferencia en tiempo real ───────────────────────────────────

SNIPPET_INFERENCIA = '''"""
inferencia_lsm.py  (generado automáticamente)
----------------------------------------------
Carga el modelo entrenado y clasifica en tiempo real con MediaPipe.
"""
import joblib
import numpy as np

# Cargar artefactos
pipeline = joblib.load("data/processed/mejor_modelo.pkl")
le       = joblib.load("data/processed/label_encoder.pkl")

def predecir_sena(landmarks_izq, landmarks_der):
    """
    landmarks_izq / landmarks_der: lista de 63 floats normalizados
                                   (salida de normalizar_mano), o None.
    Retorna: (seña_predicha: str, confianza: float)
    """
    if landmarks_izq is None and landmarks_der is None:
        return None, 0.0

    # Usar la mano disponible (prioridad: izquierda)
    vec = np.array(landmarks_izq if landmarks_izq is not None else landmarks_der,
                   dtype=np.float32).reshape(1, -1)

    proba  = pipeline.predict_proba(vec)[0]
    idx    = np.argmax(proba)
    sena   = le.inverse_transform([idx])[0]
    conf   = float(proba[idx])
    return sena, conf


# ── Ejemplo de uso en bucle de cámara ──────────────────────────────────────
# while True:
#     ...detectar landmarks con MediaPipe...
#     sena, conf = predecir_sena(mano_izq_normalizada, mano_der_normalizada)
#     if conf > 0.75:
#         print(f"Seña: {sena}  ({conf:.1%})")
'''


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n🚀  ENTRENAMIENTO DEL CLASIFICADOR LSM")
    print("=" * 55)

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    # 1. Cargar y limpiar CSV
    df_estatico = cargar_y_limpiar_csv()

    # 2. Cargar secuencias (J, K, Z)
    df_secuencias = cargar_secuencias()

    # 3. Preparar datos
    X_train, X_val, X_test, y_train, y_val, y_test, le = preparar_datos(
        df_estatico, df_secuencias
    )

    # 4. Entrenar y evaluar modelos
    modelos = construir_modelos()
    mejor_nombre, mejor_pipeline, resultados, acc_test, f1_test, y_pred_test = \
        entrenar_y_evaluar(modelos, X_train, X_val, X_test, y_train, y_val, y_test, le)

    # 5. Visualizaciones
    print("\n" + "=" * 55)
    print("5. GENERANDO VISUALIZACIONES")
    print("=" * 55)
    graficar_confusion(y_test, y_pred_test, le, mejor_nombre)
    graficar_comparacion_modelos(resultados)

    # 6. Guardar artefactos
    guardar_artefactos(mejor_pipeline, le, mejor_nombre, resultados, acc_test, f1_test)

    # 7. Generar snippet de inferencia
    ruta_snippet = DIR_SALIDA / "inferencia_lsm.py"
    ruta_snippet.write_text(SNIPPET_INFERENCIA, encoding="utf-8")
    print(f"   Snippet inferencia: {ruta_snippet}")

    # Resumen final
    print("\n" + "=" * 55)
    print("✅  ENTRENAMIENTO COMPLETADO")
    print("=" * 55)
    print(f"   Modelo ganador: {mejor_nombre}")
    print(f"   Accuracy  test: {acc_test:.2%}")
    print(f"   F1 weighted:    {f1_test:.2%}")
    print(f"   Clases ({len(le.classes_)}): {list(le.classes_)}")
    print("\n   Siguientes pasos:")
    print("   1. Revisar confusion_matrix.png para ver señas difíciles")
    print("   2. Completar secuencias J/K/Z hasta 50 muestras cada una")
    print("   3. Integrar inferencia_lsm.py en collect_dataset.py para demo")
    print("=" * 55)


if __name__ == "__main__":
    main()