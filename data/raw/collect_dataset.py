"""

DETECTA AMBAS MANOS - CON SOPORTE PARA MOVIMIENTOS (SECUENCIAS)



Controles:
    ESPACIO  -> capturar muestra (foto estática)
    M        -> capturar secuencia (3 segundos de video) para movimientos
    N        -> siguiente sena
    Q        -> guardar y salir
"""

import csv
import time
import json
from pathlib import Path
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request

# ─── Configuracion ────────────────────────────────────────────────────────────

SENAS = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z"
]

# Señales que requieren movimiento (captura de secuencia)
SENAS_CON_MOVIMIENTO = ["J", "Z", "K"]  # Puedes agregar más según necesites

MUESTRAS_X_SENA = 50
ARCHIVO_CSV     = Path("data/processed/dataset_lsm.csv")
ARCHIVO_SECUENCIAS = Path("data/processed/secuencias_lsm.json")
COOLDOWN_SEG    = 0.4
MODELO_PATH     = Path("models/hand_landmarker.task")
DURACION_SECUENCIA = 3.0  # 3 segundos
FPS_MUESTREO = 10  # Muestras por segundo durante la secuencia

# ─── Colores BGR ─────────────────────────────────────────────────────────────
VERDE  = (0,  210, 120)
ROJO   = (50,  50, 210)
BLANCO = (255, 255, 255)
GRIS   = (130, 130, 130)
CIAN   = (0,  220, 180)
NEGRO  = (20,  20,  20)
AMARILLO = (0, 255, 255)  # Para mano izquierda
MORADO = (255, 0, 255)    # Para mano derecha
NARANJA = (0, 165, 255)   # Para modo secuencia

# Inicializar drawing_utils (solo para visualización)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# ─── Descargar modelo si no existe ───────────────────────────────────────────

def descargar_modelo():
    MODELO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODELO_PATH.exists():
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        print("[INFO] Descargando modelo de MediaPipe (~25 MB)...")
        urllib.request.urlretrieve(url, MODELO_PATH)
        print(f"[INFO] Modelo guardado en: {MODELO_PATH}")
    else:
        print(f"[INFO] Modelo encontrado: {MODELO_PATH}")

# ─── Normalizar landmarks ─────────────────────────────────────────────────────

def normalizar_mano(landmarks):
    """
    Normaliza los landmarks de una mano con respecto a su muñeca
    Retorna lista con 63 valores (21 puntos * 3 coordenadas)
    """
    raw = []
    for lm in landmarks:
        raw.extend([lm.x, lm.y, lm.z])
    
    # Usar el punto 0 (muñeca) como referencia
    x0, y0, z0 = raw[0], raw[1], raw[2]
    resultado = []
    for i in range(0, len(raw), 3):
        resultado.extend([
            round(raw[i]   - x0, 6),
            round(raw[i+1] - y0, 6),
            round(raw[i+2] - z0, 6),
        ])
    
    return resultado

def combinar_landmarks(mano_izq, mano_der):
    """Combina los landmarks de ambas manos en un solo vector"""
    if mano_izq is None and mano_der is None:
        return None
    
    resultado = []
    
    # Primero la mano izquierda (63 valores)
    if mano_izq:
        resultado.extend(mano_izq)
    else:
        resultado.extend([0.0] * 63)  # 21*3 = 63 ceros
    
    # Luego la mano derecha (63 valores)
    if mano_der:
        resultado.extend(mano_der)
    else:
        resultado.extend([0.0] * 63)
    
    return resultado

# ─── CSV para muestras estáticas ────────────────────────────────────────────

def inicializar_csv():
    ARCHIVO_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVO_CSV.exists():
        encabezado = ["etiqueta"]
        
        # Columnas para mano izquierda (21 puntos x 3 coordenadas)
        for i in range(21):
            encabezado += [f"izq_x{i}", f"izq_y{i}", f"izq_z{i}"]
        
        # Columnas para mano derecha (21 puntos x 3 coordenadas)
        for i in range(21):
            encabezado += [f"der_x{i}", f"der_y{i}", f"der_z{i}"]
        
        with open(ARCHIVO_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(encabezado)
        print(f"[INFO] CSV creado: {ARCHIVO_CSV}")
    else:
        print(f"[INFO] CSV existente: {ARCHIVO_CSV}")

def contar_muestras():
    conteo = {s: 0 for s in SENAS}
    if ARCHIVO_CSV.exists():
        with open(ARCHIVO_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lbl = row.get("etiqueta", "").strip()
                if lbl in conteo:
                    conteo[lbl] += 1
    return conteo

def guardar_muestra(etiqueta, landmarks_combinados):
    with open(ARCHIVO_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([etiqueta] + landmarks_combinados)

# ─── JSON para secuencias (movimientos) ─────────────────────────────────────

def inicializar_json_secuencias():
    ARCHIVO_SECUENCIAS.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVO_SECUENCIAS.exists():
        with open(ARCHIVO_SECUENCIAS, "w", encoding="utf-8") as f:
            json.dump({}, f)
        print(f"[INFO] Archivo de secuencias creado: {ARCHIVO_SECUENCIAS}")
    else:
        print(f"[INFO] Archivo de secuencias existente: {ARCHIVO_SECUENCIAS}")

def contar_secuencias():
    """Cuenta cuántas secuencias hay por cada seña"""
    conteo = {s: 0 for s in SENAS}
    if ARCHIVO_SECUENCIAS.exists():
        with open(ARCHIVO_SECUENCIAS, "r", encoding="utf-8") as f:
            data = json.load(f)
            for sena, secuencias in data.items():
                if sena in conteo:
                    conteo[sena] = len(secuencias)
    return conteo

def guardar_secuencia(etiqueta, secuencia_frames):
    """Guarda una secuencia de frames (lista de landmarks por frame)"""
    with open(ARCHIVO_SECUENCIAS, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if etiqueta not in data:
        data[etiqueta] = []
    
    # Agregar nueva secuencia con timestamp
    secuencia_info = {
        "timestamp": datetime.now().isoformat(),
        "frames": secuencia_frames,
        "num_frames": len(secuencia_frames)
    }
    data[etiqueta].append(secuencia_info)
    
    with open(ARCHIVO_SECUENCIAS, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ─── Captura de secuencia ────────────────────────────────────────────────────

def capturar_secuencia(detector, cap, duracion=3.0, fps_muestreo=10):
    """
    Captura una secuencia de landmarks durante 'duracion' segundos
    Retorna lista de landmarks por frame
    """
    print(f"[INFO] Preparando captura de secuencia de {duracion} segundos...")
    time.sleep(1)  # Dar tiempo para prepararse
    
    frames_landmarks = []
    inicio = time.time()
    ultimo_muestreo = inicio
    frame_count = 0
    
    while time.time() - inicio < duracion:
        ok, frame = cap.read()
        if not ok:
            break
        
        # Procesar frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        resultado = detector.detect(mp_image)
        
        # Extraer landmarks de ambas manos
        mano_usuario_izq = None
        mano_usuario_der = None
        
        if resultado.hand_landmarks:
            for hand_landmarks, handedness in zip(resultado.hand_landmarks, resultado.handedness):
                mano_normalizada = normalizar_mano(hand_landmarks)
                
                # Intercambiar clasificación por el espejo
                if handedness[0].category_name == "Left":
                    mano_usuario_der = mano_normalizada
                else:
                    mano_usuario_izq = mano_normalizada
        
        # Combinar landmarks
        landmarks_combinados = combinar_landmarks(mano_usuario_izq, mano_usuario_der)
        
        # Muestrear a frecuencia específica
        ahora = time.time()
        if ahora - ultimo_muestreo >= 1.0 / fps_muestreo:
            if landmarks_combinados:
                frames_landmarks.append(landmarks_combinados)
            else:
                # Si no hay manos, agregar ceros
                frames_landmarks.append([0.0] * 126)  # 63+63 = 126
            ultimo_muestreo = ahora
            frame_count += 1
            
            # Mostrar progreso
            cv2.putText(frame, f"Capturando secuencia... {int(ahora - inicio)}/{duracion}s", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, NARANJA, 2)
        
        cv2.imshow("LSM — Capturando secuencia", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"[INFO] Secuencia capturada: {len(frames_landmarks)} frames")
    return frames_landmarks

# ─── Dibujar landmarks con colores diferenciados ─────────────────────────────

def dibujar_landmarks(frame, hand_landmarks_list, handedness_list):
    """Dibuja landmarks de ambas manos con colores diferentes"""
    if not hand_landmarks_list:
        return
    
    for idx, (hand_landmarks, handedness) in enumerate(zip(hand_landmarks_list, handedness_list)):
        hand_label = handedness[0].category_name
        
        if hand_label == "Left":
            color_puntos = AMARILLO
            color_lineas = (0, 165, 255)
        else:
            color_puntos = MORADO
            color_lineas = (255, 0, 128)
        
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            
            h, w = frame.shape[:2]
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(frame, start_point, end_point, color_lineas, 2)
        
        for lm in hand_landmarks:
            h, w = frame.shape[:2]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, color_puntos, -1)
        
        h, w = frame.shape[:2]
        wrist = hand_landmarks[0]
        cx, cy = int(wrist.x * w), int(wrist.y * h)
        cv2.putText(frame, hand_label, (cx - 20, cy - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANCO, 1)

# ─── Overlay ──────────────────────────────────────────────────────────────────

def dibujar_overlay(frame, sena, idx, total, capturadas, objetivo, 
                    manos_detectadas, flash, es_movimiento=False):
    h, w = frame.shape[:2]
    num_manos = len(manos_detectadas) if manos_detectadas else 0

    cv2.rectangle(frame, (0, 0), (w, 52), NEGRO, -1)
    
    # Indicar si es seña con movimiento
    if es_movimiento:
        modo_texto = "MODO MOVIMIENTO (Secuencia) - Presiona M"
        color_modo = NARANJA
    else:
        modo_texto = "MODO ESTATICO - Presiona ESPACIO"
        color_modo = BLANCO
    
    cv2.putText(frame, f"LSM Dataset  |  Sena: {sena}  ({idx+1}/{total})",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_modo, 1)

    barra_w = w - 24
    progreso = int(barra_w * capturadas / objetivo)
    cv2.rectangle(frame, (12, 58), (12 + barra_w, 72), (50, 50, 50), -1)
    cv2.rectangle(frame, (12, 58), (12 + progreso, 72), VERDE, -1)
    cv2.putText(frame, f"{capturadas}/{objetivo}",
                (14, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRIS, 1)

    if num_manos == 2:
        msg = "Ambas manos detectadas"
        color = VERDE
    elif num_manos == 1:
        msg = "Solo 1 mano detectada"
        color = AMARILLO
    else:
        msg = "Sin manos — acerca ambas manos"
        color = ROJO
    
    cv2.putText(frame, msg, (12, h - 84), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
    cv2.putText(frame, f"Manos detectadas: {num_manos}/2",
                (12, h - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRIS, 1)
    
    if es_movimiento:
        cv2.putText(frame, "M: capturar secuencia (3s)   N: siguiente   Q: salir",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, NARANJA, 1)
    else:
        cv2.putText(frame, "ESPACIO: capturar foto   M: modo movimiento   N: siguiente   Q: salir",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRIS, 1)

    if flash:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), CIAN, -1)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
        cv2.putText(frame, "CAPTURADO", (w//2 - 90, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, BLANCO, 2)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    descargar_modelo()
    inicializar_csv()
    inicializar_json_secuencias()
    
    conteo_estatico = contar_muestras()
    conteo_secuencias = contar_secuencias()
    
    # Usar el conteo apropiado según el tipo de seña
    senas_pendientes = []
    for s in SENAS:
        if s in SENAS_CON_MOVIMIENTO:
            if conteo_secuencias[s] < MUESTRAS_X_SENA:
                senas_pendientes.append(s)
        else:
            if conteo_estatico[s] < MUESTRAS_X_SENA:
                senas_pendientes.append(s)
    
    if not senas_pendientes:
        print("[INFO] Todas las senas completas.")
        return

    print(f"[INFO] Senas pendientes: {senas_pendientes}")
    print(f"[INFO] Nota: Las señas {SENAS_CON_MOVIMIENTO} requieren captura de movimiento (secuencia de 3s)\n")

    base_options = mp_python.BaseOptions(model_asset_path=str(MODELO_PATH))
    opciones = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    detector = vision.HandLandmarker.create_from_options(opciones)

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la camara.")
        return

    idx = 0
    ultimo_cap = 0.0
    flash_ts = 0.0

    while idx < len(senas_pendientes):
        sena_actual = senas_pendientes[idx]
        es_movimiento = sena_actual in SENAS_CON_MOVIMIENTO
        
        if es_movimiento:
            capturadas = conteo_secuencias[sena_actual]
        else:
            capturadas = conteo_estatico[sena_actual]

        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        resultado = detector.detect(mp_image)

        mano_usuario_izq = None
        mano_usuario_der = None
        manos_detectadas = resultado.hand_landmarks if resultado.hand_landmarks else []
        
        if manos_detectadas:
            for hand_landmarks, handedness in zip(resultado.hand_landmarks, resultado.handedness):
                mano_normalizada = normalizar_mano(hand_landmarks)
                
                if handedness[0].category_name == "Left":
                    mano_usuario_der = mano_normalizada
                else:
                    mano_usuario_izq = mano_normalizada
            
            visual_handedness = []
            for handedness in resultado.handedness:
                if handedness[0].category_name == "Left":
                    new_handedness = [type(handedness[0])()]
                    new_handedness[0].category_name = "Right"
                    visual_handedness.append(new_handedness)
                else:
                    new_handedness = [type(handedness[0])()]
                    new_handedness[0].category_name = "Left"
                    visual_handedness.append(new_handedness)
            
            dibujar_landmarks(frame, resultado.hand_landmarks, visual_handedness)

        landmarks_combinados = combinar_landmarks(mano_usuario_izq, mano_usuario_der)
        mano_ok = (mano_usuario_izq is not None or mano_usuario_der is not None)
        
        flash = (time.time() - flash_ts) < 0.2
        dibujar_overlay(frame, sena_actual, idx, len(senas_pendientes),
                        capturadas, MUESTRAS_X_SENA, manos_detectadas, flash, es_movimiento)

        cv2.imshow("LSM — Recoleccion de Dataset", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord(' '):
            if es_movimiento:
                print(f"[INFO] Para '{sena_actual}' usa M (modo movimiento)")
            else:
                ahora = time.time()
                if not mano_ok:
                    print(f"[AVISO] Sin manos para '{sena_actual}'")
                elif ahora - ultimo_cap < COOLDOWN_SEG:
                    pass
                elif capturadas >= MUESTRAS_X_SENA:
                    print(f"[INFO] '{sena_actual}' completa. Presiona N.")
                else:
                    guardar_muestra(sena_actual, landmarks_combinados)
                    conteo_estatico[sena_actual] += 1
                    capturadas += 1
                    ultimo_cap = ahora
                    flash_ts = ahora
                    
                    info_manos = []
                    if mano_usuario_izq: info_manos.append("Izquierda")
                    if mano_usuario_der: info_manos.append("Derecha")
                    print(f"[OK] {sena_actual}: {capturadas}/{MUESTRAS_X_SENA} - Manos: {', '.join(info_manos)}")
                    
                    if capturadas >= MUESTRAS_X_SENA:
                        print(f"[INFO] '{sena_actual}' completada. Presiona N.")

        elif tecla == ord('m') or tecla == ord('M'):
            if not es_movimiento:
                print(f"[INFO] '{sena_actual}' no requiere movimiento. Usa ESPACIO.")
            else:
                ahora = time.time()
                if ahora - ultimo_cap < COOLDOWN_SEG:
                    pass
                elif capturadas >= MUESTRAS_X_SENA:
                    print(f"[INFO] '{sena_actual}' completa. Presiona N.")
                else:
                    print(f"[INFO] Capturando secuencia para '{sena_actual}'...")
                    secuencia = capturar_secuencia(detector, cap, DURACION_SECUENCIA, FPS_MUESTREO)
                    if secuencia:
                        guardar_secuencia(sena_actual, secuencia)
                        conteo_secuencias[sena_actual] += 1
                        capturadas += 1
                        ultimo_cap = ahora
                        flash_ts = ahora
                        print(f"[OK] Secuencia {capturadas}/{MUESTRAS_X_SENA} guardada para '{sena_actual}'")
                    else:
                        print(f"[ERROR] No se pudo capturar secuencia")

        elif tecla in (ord('n'), ord('N')):
            if capturadas < MUESTRAS_X_SENA:
                print(f"[AVISO] '{sena_actual}' incompleta ({capturadas}/{MUESTRAS_X_SENA})")
            idx += 1

        elif tecla in (ord('q'), ord('Q')):
            print("\n[INFO] Sesion terminada.")
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print("\n" + "="*45)
    print("RESUMEN DEL DATASET")
    print("="*45)
    print("\nMuestras estáticas (fotos):")
    total = 0
    for s in SENAS:
        n = conteo_estatico[s]
        total += n
        barra = "█" * (n * 20 // MUESTRAS_X_SENA)
        estado = "OK" if n >= MUESTRAS_X_SENA else f"faltan {MUESTRAS_X_SENA - n}"
        print(f"  {s}: {n:>3}/{MUESTRAS_X_SENA}  {barra}  {estado}")
    
    print("\nSecuencias (movimientos):")
    total_sec = 0
    for s in SENAS:
        if s in SENAS_CON_MOVIMIENTO:
            n = conteo_secuencias[s]
            total_sec += n
            barra = "█" * (n * 20 // MUESTRAS_X_SENA)
            estado = "OK" if n >= MUESTRAS_X_SENA else f"faltan {MUESTRAS_X_SENA - n}"
            print(f"  {s}: {n:>3}/{MUESTRAS_X_SENA}  {barra}  {estado}")
    
    print(f"\n  Total estáticas: {total} muestras")
    print(f"  Total secuencias: {total_sec} muestras")
    print(f"  Archivo estático: {ARCHIVO_CSV}")
    print(f"  Archivo secuencias: {ARCHIVO_SECUENCIAS}")
    print("="*45)


if __name__ == "__main__":
    main()