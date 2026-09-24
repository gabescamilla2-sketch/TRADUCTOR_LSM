"""


Controles:
    Q / ESC  -> salir
    C        -> limpiar texto acumulado
    ESPACIO  -> confirmar seña actual al texto
    B        -> borrar última letra del texto

"""

import time
import collections
from pathlib import Path

import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ─── Rutas ────────────────────────────────────────────────────────────────────

MODELO_PATH    = Path("models/hand_landmarker.task")
PIPELINE_PATH  = Path("data/processed/mejor_modelo.pkl")
ENCODER_PATH   = Path("data/processed/label_encoder.pkl")

# ─── Parámetros de inferencia ─────────────────────────────────────────────────

CONFIANZA_MIN      = 0.70   # umbral mínimo para mostrar predicción
VENTANA_VOTOS      = 10     # últimas N predicciones para votación mayoritaria
COOLDOWN_CONFIRMAR = 1.2    # segundos de estabilidad antes de auto-confirmar
MIN_ESTABLE        = 8      # de las últimas N, cuántas deben coincidir para confirmar

# ─── Colores BGR ──────────────────────────────────────────────────────────────

VERDE   = (0,  210, 120)
ROJO    = (50,  50, 210)
BLANCO  = (255, 255, 255)
GRIS    = (130, 130, 130)
NEGRO   = (15,  15,  15)
CIAN    = (0,  220, 180)
AMARILLO= (0,  230, 230)
NARANJA = (0,  165, 255)
MORADO  = (200,  80, 200)

mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands

# ─── Normalización (igual que en collect_dataset.py) ─────────────────────────

def normalizar_mano(landmarks) -> list:
    """Normaliza landmarks de una mano respecto a la muñeca. Retorna 63 floats."""
    raw = []
    for lm in landmarks:
        raw.extend([lm.x, lm.y, lm.z])
    x0, y0, z0 = raw[0], raw[1], raw[2]
    result = []
    for i in range(0, len(raw), 3):
        result.extend([
            round(raw[i]   - x0, 6),
            round(raw[i+1] - y0, 6),
            round(raw[i+2] - z0, 6),
        ])
    return result

# ─── Predicción ───────────────────────────────────────────────────────────────

def predecir(pipeline, le, landmarks_izq, landmarks_der):
    """
    Clasifica la seña a partir de los landmarks de una o ambas manos.
    Prioridad: mano izquierda; si no hay, usa la derecha.
    Retorna (seña: str | None, confianza: float, probas: np.ndarray)
    """
    if landmarks_izq is None and landmarks_der is None:
        return None, 0.0, None

    vec = np.array(
        landmarks_izq if landmarks_izq is not None else landmarks_der,
        dtype=np.float32
    )

    # El pipeline fue entrenado con 441 features (63 estáticos + 378 padding)
    # Para inferencia estática solo necesitamos los 63, con padding de ceros
    n_features = pipeline.named_steps["clf"].n_features_in_
    if len(vec) < n_features:
        vec = np.concatenate([vec, np.zeros(n_features - len(vec), dtype=np.float32)])

    vec = vec.reshape(1, -1)

    probas = pipeline.predict_proba(vec)[0]
    idx    = int(np.argmax(probas))
    conf   = float(probas[idx])
    sena   = le.inverse_transform([idx])[0]

    return sena, conf, probas

# ─── Overlay de UI ────────────────────────────────────────────────────────────

def dibujar_landmarks(frame, hand_landmarks_list, handedness_list):
    """Dibuja landmarks con color según la mano."""
    for hand_lms, handedness in zip(hand_landmarks_list, handedness_list):
        color = AMARILLO if handedness[0].category_name == "Left" else MORADO
        h, w = frame.shape[:2]
        for lm in hand_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, color, -1)
        # Conexiones
        connections = mp_hands.HAND_CONNECTIONS
        for conn in connections:
            a, b = conn
            ax_ = int(hand_lms[a].x * w); ay_ = int(hand_lms[a].y * h)
            bx_ = int(hand_lms[b].x * w); by_ = int(hand_lms[b].y * h)
            cv2.line(frame, (ax_, ay_), (bx_, by_), color, 1)


def barra_confianza(frame, x, y, confianza, ancho=160, alto=10):
    """Dibuja una barra de confianza horizontal."""
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (60, 60, 60), -1)
    fill = int(ancho * confianza)
    color = VERDE if confianza >= CONFIANZA_MIN else NARANJA
    cv2.rectangle(frame, (x, y), (x + fill, y + alto), color, -1)
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), GRIS, 1)


def dibujar_overlay(frame, sena_pred, confianza, votos, texto_acumulado,
                    estabilidad, num_manos, top3, le):
    h, w = frame.shape[:2]

    # ── Panel izquierdo (semitransparente) ────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (230, h), NEGRO, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Título
    cv2.putText(frame, "LSM", (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, CIAN, 2)
    cv2.putText(frame, "Traductor", (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRIS, 1)
    cv2.line(frame, (14, 68), (215, 68), (60, 60, 60), 1)

    # Seña predicha
    cv2.putText(frame, "PREDICCION", (14, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRIS, 1)

    if sena_pred and confianza >= CONFIANZA_MIN:
        color_sena = VERDE
        texto_sena = sena_pred
    elif sena_pred:
        color_sena = NARANJA
        texto_sena = sena_pred
    else:
        color_sena = GRIS
        texto_sena = "---"

    cv2.putText(frame, texto_sena, (14, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 3.2, color_sena, 5)

    # Barra de confianza
    cv2.putText(frame, f"{confianza:.0%}", (14, 178),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRIS, 1)
    barra_confianza(frame, 60, 168, confianza)

    # Estabilidad (cuántas de las últimas N coinciden)
    cv2.putText(frame, "Estabilidad", (14, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRIS, 1)
    pct_est = estabilidad / VENTANA_VOTOS if VENTANA_VOTOS > 0 else 0
    barra_confianza(frame, 14, 210, pct_est, ancho=200)
    cv2.putText(frame, f"{estabilidad}/{VENTANA_VOTOS}", (14, 234),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRIS, 1)

    # Indicador de auto-confirmación
    if estabilidad >= MIN_ESTABLE and sena_pred:
        cv2.putText(frame, f">> auto-confirmando...", (14, 255),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, AMARILLO, 1)

    cv2.line(frame, (14, 270), (215, 270), (60, 60, 60), 1)

    # Top 3 candidatos
    cv2.putText(frame, "TOP 3", (14, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRIS, 1)
    if top3 is not None:
        for rank, (sena_t, prob_t) in enumerate(top3):
            y_t = 310 + rank * 22
            color_t = BLANCO if rank == 0 else GRIS
            cv2.putText(frame, f"{rank+1}. {sena_t}  {prob_t:.0%}",
                        (14, y_t), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_t, 1)
            barra_confianza(frame, 100, y_t - 12, prob_t, ancho=110, alto=7)

    cv2.line(frame, (14, 380), (215, 380), (60, 60, 60), 1)

    # Estado de manos
    cv2.putText(frame, f"Manos: {num_manos}", (14, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRIS, 1)

    # ── Panel inferior (texto acumulado) ──────────────────────────────────────
    panel_y = h - 70
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, panel_y), (w, h), NEGRO, -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "Texto:", (10, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRIS, 1)

    # Mostrar solo los últimos 30 caracteres si es muy largo
    texto_display = texto_acumulado[-30:] if len(texto_acumulado) > 30 else texto_acumulado
    if len(texto_acumulado) > 30:
        texto_display = "..." + texto_display

    cv2.putText(frame, texto_display if texto_display else "_",
                (70, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                CIAN if texto_acumulado else GRIS, 2)

    # Controles
    ctrl = "ESPACIO:confirmar  B:borrar  C:limpiar  Q:salir"
    cv2.putText(frame, ctrl, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, GRIS, 1)

    # ── Línea separadora entre panel y cámara ────────────────────────────────
    cv2.line(frame, (230, 0), (230, h), (60, 60, 60), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Verificar archivos necesarios
    for ruta in [PIPELINE_PATH, ENCODER_PATH, MODELO_PATH]:
        if not ruta.exists():
            print(f"[ERROR] No encontrado: {ruta}")
            if ruta == MODELO_PATH:
                print("        Ejecuta collect_dataset.py primero (descarga el modelo).")
            elif "pkl" in str(ruta):
                print("        Ejecuta entrenar_modelo.py primero.")
            return

    # Cargar modelo
    print("[INFO] Cargando modelo...")
    pipeline = joblib.load(PIPELINE_PATH)
    le       = joblib.load(ENCODER_PATH)
    n_clases = len(le.classes_)
    print(f"[INFO] Modelo MLP cargado | {n_clases} clases: {list(le.classes_)}")

    # Inicializar detector MediaPipe
    base_options = mp_python.BaseOptions(model_asset_path=str(MODELO_PATH))
    opciones = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.55,
    )
    detector = vision.HandLandmarker.create_from_options(opciones)

    # Cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # Estado
    historial     = collections.deque(maxlen=VENTANA_VOTOS)
    texto_acum    = ""
    ultima_conf   = 0.0
    ultimo_t_estable = None
    sena_estable  = None
    ultima_auto   = ""   # evitar confirmar la misma seña dos veces seguidas

    print("[INFO] Iniciando inferencia. Controles:")
    print("       ESPACIO → confirmar seña | B → borrar | C → limpiar | Q/ESC → salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = detector.detect(mp_img)

        mano_izq = None
        mano_der = None
        num_manos = 0

        if resultado.hand_landmarks:
            num_manos = len(resultado.hand_landmarks)
            for hand_lms, handedness in zip(resultado.hand_landmarks, resultado.handedness):
                norm = normalizar_mano(hand_lms)
                # MediaPipe espeja → Left en cámara = mano derecha del usuario
                if handedness[0].category_name == "Left":
                    mano_der = norm
                else:
                    mano_izq = norm
            dibujar_landmarks(frame, resultado.hand_landmarks, resultado.handedness)

        # Predicción
        sena_pred, confianza, probas = predecir(pipeline, le, mano_izq, mano_der)
        ultima_conf = confianza

        # Historial para votación mayoritaria
        if sena_pred and confianza >= CONFIANZA_MIN:
            historial.append(sena_pred)
        else:
            historial.append(None)

        # Calcular voto mayoritario y estabilidad
        if historial:
            conteo = collections.Counter(v for v in historial if v is not None)
            if conteo:
                sena_voto, voto_count = conteo.most_common(1)[0]
            else:
                sena_voto, voto_count = None, 0
        else:
            sena_voto, voto_count = None, 0

        # Top 3 para mostrar en UI
        top3 = None
        if probas is not None:
            idx_sorted = np.argsort(probas)[::-1][:3]
            top3 = [(le.inverse_transform([i])[0], float(probas[i])) for i in idx_sorted]

        # Auto-confirmación por estabilidad
        ahora = time.time()
        if sena_voto and voto_count >= MIN_ESTABLE:
            if sena_voto != sena_estable:
                sena_estable    = sena_voto
                ultimo_t_estable = ahora
            elif ahora - ultimo_t_estable >= COOLDOWN_CONFIRMAR:
                if sena_voto != ultima_auto:  # no repetir automáticamente
                    texto_acum += sena_voto
                    ultima_auto = sena_voto
                    ultimo_t_estable = ahora
                    print(f"[AUTO] Confirmado: {sena_voto}  → \"{texto_acum}\"")
        else:
            sena_estable    = None
            ultimo_t_estable = None

        # Dibujar UI
        dibujar_overlay(
            frame,
            sena_pred=sena_voto if sena_voto else sena_pred,
            confianza=confianza,
            votos=voto_count,
            texto_acumulado=texto_acum,
            estabilidad=voto_count,
            num_manos=num_manos,
            top3=top3,
            le=le
        )

        cv2.imshow("LSM — Inferencia en Tiempo Real", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla in (ord('q'), ord('Q'), 27):  # Q o ESC
            break

        elif tecla == ord(' '):  # Confirmar manualmente
            if sena_voto and confianza >= CONFIANZA_MIN:
                texto_acum += sena_voto
                ultima_auto = sena_voto
                print(f"[MANUAL] Confirmado: {sena_voto}  → \"{texto_acum}\"")
            elif sena_pred:
                print(f"[AVISO] Confianza baja ({confianza:.0%}), no confirmado.")

        elif tecla in (ord('b'), ord('B')):  # Borrar última letra
            if texto_acum:
                texto_acum = texto_acum[:-1]
                print(f"[BORRAR] → \"{texto_acum}\"")

        elif tecla in (ord('c'), ord('C')):  # Limpiar todo
            texto_acum = ""
            ultima_auto = ""
            print("[LIMPIAR] Texto borrado.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print(f"\n[FIN] Texto acumulado: \"{texto_acum}\"")


if __name__ == "__main__":
    main()