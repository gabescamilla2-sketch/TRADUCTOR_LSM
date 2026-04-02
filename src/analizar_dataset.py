"""
analizar_dataset.py
-------------------
Análisis y preprocesamiento del dataset de LSM
(CORREGIDO - Maneja datos nulos y visualización)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuración
ARCHIVO_CSV = Path("data/processed/dataset_lsm.csv")
ARCHIVO_SECUENCIAS = Path("data/processed/secuencias_lsm.json")
SENAS = ["A","B","C","D","E","F","G","H","I","J",
         "K","L","M","N","O","P","Q","R","S","T",
         "U","V","W","X","Y","Z"]

def analizar_dataset_estatico():
    """Analiza el dataset de muestras estáticas"""
    print("="*50)
    print("ANÁLISIS DE DATOS ESTÁTICOS")
    print("="*50)
    
    # Cargar datos
    df = pd.read_csv(ARCHIVO_CSV)
    
    # Limpiar datos nulos en la columna etiqueta
    df_clean = df.dropna(subset=['etiqueta'])
    filas_nulas = len(df) - len(df_clean)
    
    print(f"\n1. Información general:")
    print(f"   - Total de muestras: {len(df)}")
    print(f"   - Filas con etiqueta nula: {filas_nulas}")
    print(f"   - Muestras válidas: {len(df_clean)}")
    print(f"   - Columnas: {len(df.columns)}")
    print(f"   - Señas únicas: {df_clean['etiqueta'].nunique()}")
    
    if filas_nulas > 0:
        print(f"\n⚠️ Advertencia: Se encontraron {filas_nulas} filas con etiqueta nula")
        print("   Estas filas serán ignoradas en el análisis")
    
    print(f"\n2. Distribución por seña:")
    distribucion = df_clean['etiqueta'].value_counts().sort_index()
    for sena, count in distribucion.items():
        print(f"   - {sena}: {count} muestras")
    
    # Verificar señas faltantes
    senas_en_dataset = set(distribucion.index)
    senas_faltantes = set(SENAS) - senas_en_dataset
    if senas_faltantes:
        print(f"\n   Señas faltantes (deben usar secuencias): {senas_faltantes}")
    
    # Graficar distribución
    plt.figure(figsize=(14, 6))
    distribucion.plot(kind='bar')
    plt.title('Distribución de muestras por seña (estático)')
    plt.xlabel('Seña')
    plt.ylabel('Número de muestras')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/processed/distribucion_estatico.png')
    print(f"\n   Gráfico guardado: data/processed/distribucion_estatico.png")
    
    # Análisis de landmarks
    landmarks_cols = [col for col in df.columns if col != 'etiqueta']
    print(f"\n3. Análisis de landmarks:")
    print(f"   - Total de features: {len(landmarks_cols)}")
    print(f"   - Features por mano: {len(landmarks_cols)//2}")
    
    return df_clean

def analizar_secuencias():
    """Analiza las secuencias de movimiento"""
    print("\n" + "="*50)
    print("ANÁLISIS DE SECUENCIAS")
    print("="*50)
    
    if not ARCHIVO_SECUENCIAS.exists():
        print("No hay archivo de secuencias")
        return None
    
    with open(ARCHIVO_SECUENCIAS, 'r') as f:
        data = json.load(f)
    
    print(f"\n1. Información general:")
    total_secuencias = 0
    for sena, secuencias in data.items():
        print(f"   - {sena}: {len(secuencias)} secuencias")
        total_secuencias += len(secuencias)
        if secuencias:
            frames = secuencias[0].get('frames', [])
            print(f"     * Frames por secuencia: {len(frames)}")
            print(f"     * Features por frame: {len(frames[0]) if frames else 0}")
    
    print(f"\n   Total de secuencias: {total_secuencias}")
    
    return data

def visualizar_landmarks(df):
    """Visualización de los landmarks promedio por seña"""
    print("\n" + "="*50)
    print("VISUALIZACIÓN DE LANDMARKS")
    print("="*50)
    
    # Seleccionar algunas señas para visualizar
    senas_ejemplo = ['A', 'B', 'C', 'L', 'Y']
    senas_disponibles = [s for s in senas_ejemplo if s in df['etiqueta'].values and len(df[df['etiqueta'] == s]) > 0]
    
    if not senas_disponibles:
        print("No hay suficientes datos para visualizar")
        return
    
    # Calcular número de subplots necesarios
    n_plots = len(senas_disponibles)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, sena in enumerate(senas_disponibles):
        # Obtener datos de la seña
        df_sena = df[df['etiqueta'] == sena]
        
        if len(df_sena) == 0:
            continue
        
        # Separar mano izquierda y derecha
        izq_cols = [col for col in df.columns if col.startswith('izq_') and 'x' in col and col != 'etiqueta']
        der_cols = [col for col in df.columns if col.startswith('der_') and 'x' in col and col != 'etiqueta']
        
        # Tomar solo las coordenadas x de los primeros 21 puntos
        izq_cols = izq_cols[:21]  # Solo las coordenadas x
        der_cols = der_cols[:21]  # Solo las coordenadas x
        
        if len(izq_cols) == 0 or len(der_cols) == 0:
            print(f"   Advertencia: No se encontraron columnas para la seña {sena}")
            continue
        
        izq_mean = df_sena[izq_cols].mean().values
        der_mean = df_sena[der_cols].mean().values
        
        # Verificar que los arrays no estén vacíos
        if len(izq_mean) == 0 or len(der_mean) == 0:
            print(f"   Advertencia: Datos vacíos para la seña {sena}")
            continue
        
        # Graficar
        ax = axes[idx]
        x_pos = np.arange(len(izq_mean))
        ax.bar(x_pos - 0.2, izq_mean, width=0.4, alpha=0.6, label='Mano Izq', color='orange')
        ax.bar(x_pos + 0.2, der_mean, width=0.4, alpha=0.6, label='Mano Der', color='purple')
        ax.set_title(f'Seña {sena}')
        ax.set_xlabel('Punto de referencia')
        ax.set_ylabel('Posición X normalizada')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Ocultar subplots no utilizados
    for idx in range(len(senas_disponibles), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('data/processed/landmarks_promedio.png')
    print("✅ Visualización guardada: data/processed/landmarks_promedio.png")

def preprocesar_datos_estaticos(df, test_size=0.2, val_size=0.1):
    """Preprocesa los datos para el entrenamiento"""
    print("\n" + "="*50)
    print("PREPROCESAMIENTO")
    print("="*50)
    
    # Verificar que hay suficientes muestras por clase
    distribucion = df['etiqueta'].value_counts()
    clases_minimas = distribucion[distribucion < 2].index.tolist()
    
    if clases_minimas:
        print(f"\n⚠️ Advertencia: Clases con menos de 2 muestras: {clases_minimas}")
        print("   Estas clases serán eliminadas del entrenamiento")
        df = df[~df['etiqueta'].isin(clases_minimas)]
    
    if len(df) == 0:
        print("\n❌ No hay datos suficientes para entrenamiento")
        return None
    
    # Separar features y labels
    X = df.drop('etiqueta', axis=1).values
    y = df['etiqueta'].values
    
    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n1. Datos originales:")
    print(f"   - X shape: {X.shape}")
    print(f"   - Clases: {label_encoder.classes_}")
    print(f"   - Número de clases: {len(label_encoder.classes_)}")
    
    # Dividir en entrenamiento, validación y prueba con estratificación
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Ajustar tamaño de validación
        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, 
            random_state=42, stratify=y_temp
        )
        
        print(f"\n2. División de datos:")
        print(f"   - Entrenamiento: {len(X_train)} muestras")
        print(f"   - Validación: {len(X_val)} muestras")
        print(f"   - Prueba: {len(X_test)} muestras")
        
    except ValueError as e:
        print(f"\n⚠️ Error en división estratificada: {e}")
        print("   Usando división simple sin estratificación...")
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )
        
        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, 
            random_state=42
        )
        
        print(f"\n2. División de datos (sin estratificación):")
        print(f"   - Entrenamiento: {len(X_train)} muestras")
        print(f"   - Validación: {len(X_val)} muestras")
        print(f"   - Prueba: {len(X_test)} muestras")
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else np.array([])
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])
    
    print(f"\n3. Normalización completada")
    print(f"   - Media después de normalizar: {X_train_scaled.mean():.2e}")
    print(f"   - Desviación después de normalizar: {X_train_scaled.std():.2f}")
    
    # Reducción de dimensionalidad opcional (PCA)
    pca = None
    try:
        if len(X_train_scaled) > 10:
            pca = PCA(n_components=0.95)  # Mantener 95% de varianza
            X_train_pca = pca.fit_transform(X_train_scaled)
            print(f"\n4. PCA (95% varianza):")
            print(f"   - Features originales: {X_train_scaled.shape[1]}")
            print(f"   - Features después de PCA: {X_train_pca.shape[1]}")
            print(f"   - Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")
        else:
            print("\n4. PCA no aplicado (insuficientes muestras)")
    except:
        print("\n4. PCA no aplicado (error en el cálculo)")
        pca = None
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'pca': pca,
        'clases': label_encoder.classes_
    }

def guardar_datos_preprocesados(data, nombre_archivo="data/processed/datos_entrenamiento.npz"):
    """Guarda los datos preprocesados"""
    if data is None:
        print("\n❌ No hay datos para guardar")
        return
    
    # Guardar arrays
    np.savez_compressed(
        nombre_archivo,
        X_train=data['X_train'],
        X_val=data['X_val'],
        X_test=data['X_test'],
        y_train=data['y_train'],
        y_val=data['y_val'],
        y_test=data['y_test'],
        clases=data['clases']
    )
    
    # Guardar scaler y pca si existen
    import joblib
    joblib.dump(data['scaler'], "data/processed/scaler.pkl")
    if data['pca']:
        joblib.dump(data['pca'], "data/processed/pca.pkl")
    
    print(f"\n✅ Datos guardados en: {nombre_archivo}")
    print(f"✅ Scaler guardado en: data/processed/scaler.pkl")
    if data['pca']:
        print(f"✅ PCA guardado en: data/processed/pca.pkl")

def generar_reporte(df, secuencias):
    """Genera un reporte completo del dataset"""
    print("\n" + "="*50)
    print("REPORTE COMPLETO DEL DATASET")
    print("="*50)
    
    # Datos estáticos
    print("\n📊 DATOS ESTÁTICOS:")
    print(f"   Total muestras: {len(df)}")
    print(f"   Señas capturadas: {df['etiqueta'].nunique()}")
    print(f"   Promedio muestras por seña: {len(df)/df['etiqueta'].nunique():.1f}")
    
    # Datos de secuencias
    if secuencias:
        total_sec = sum(len(sec) for sec in secuencias.values())
        print(f"\n🎬 DATOS DE MOVIMIENTO:")
        print(f"   Total secuencias: {total_sec}")
        print(f"   Señas con movimiento: {list(secuencias.keys())}")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        for sena, secs in secuencias.items():
            if len(secs) < 50:
                print(f"   - Para la seña '{sena}': faltan {50 - len(secs)} secuencias")
        
        # Señas que aún no tienen datos
        senas_con_mov = set(secuencias.keys())
        senas_faltantes = set(['J', 'K', 'Z']) - senas_con_mov
        if senas_faltantes:
            print(f"   - Señales sin datos de movimiento: {senas_faltantes}")
    
    # Estado general
    print(f"\n📈 ESTADO GENERAL:")
    senas_completas = df[df['etiqueta'].apply(lambda x: df['etiqueta'].value_counts()[x] >= 50)]['etiqueta'].nunique()
    print(f"   Señales estáticas completas: {senas_completas}/23")
    if secuencias:
        senas_mov_completas = sum(1 for s in secuencias if len(secuencias[s]) >= 50)
        print(f"   Señales con movimiento completas: {senas_mov_completas}/3")
    
    print(f"\n✅ Análisis completado exitosamente!")

def main():
    print("🚀 ANÁLISIS Y PREPROCESAMIENTO DEL DATASET LSM")
    print("="*50)
    
    # 1. Analizar datos estáticos
    df_estatico = analizar_dataset_estatico()
    
    # 2. Analizar secuencias
    secuencias = analizar_secuencias()
    
    # 3. Visualizar landmarks
    if len(df_estatico) > 0:
        try:
            visualizar_landmarks(df_estatico)
        except Exception as e:
            print(f"\n⚠️ Error en visualización: {e}")
            print("   Continuando con el análisis...")
    
    # 4. Preprocesar datos
    if len(df_estatico) >= 10:  # Mínimo de muestras para entrenar
        try:
            datos_prep = preprocesar_datos_estaticos(df_estatico)
            if datos_prep:
                guardar_datos_preprocesados(datos_prep)
        except Exception as e:
            print(f"\n❌ Error en preprocesamiento: {e}")
            print("   Continuando con análisis básico...")
    else:
        print(f"\n⚠️ Datos insuficientes: solo {len(df_estatico)} muestras")
        print("   Se necesitan al menos 10 muestras para entrenamiento")
    
    # 5. Generar reporte
    generar_reporte(df_estatico, secuencias)
    
    print("\n✨ ANÁLISIS COMPLETADO!")
    print("\nSiguientes pasos:")
    print("1. Completar las muestras faltantes (50 por seña)")
    print("2. Ejecutar entrenar_modelo.py")
    print("3. Probar el clasificador en tiempo real")

if __name__ == "__main__":
    main()