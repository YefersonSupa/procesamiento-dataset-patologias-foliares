"""
Script de Análisis Exploratorio de Datos (EDA) y Visualización de Calidad
Proyecto Semestral: Preprocesamiento de Dataset de Imágenes
Docente: CCACYAHUILLCA BEJAR, HANS HARLEY
Autor: SUPA CUSIPAUCAR, Yeferson - 220553
UNSAAC - 2026-2
"""

import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from preprocesamiento import auditar_calidad_imagen, realzar_contraste_clahe_lab, reducir_ruido_bilateral
except ImportError:
    from src.preprocesamiento import auditar_calidad_imagen, realzar_contraste_clahe_lab, reducir_ruido_bilateral


def generar_reporte_eda(
    dir_raw: str = None,
    dir_processed: str = None,
    salida_grafica: str = None
):
    base_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dir_raw is None:
        dir_raw = os.path.join(base_proj, 'data_raw')
    if dir_processed is None:
        dir_processed = os.path.join(base_proj, 'data_processed')
    if salida_grafica is None:
        assets_dir = os.path.join(base_proj, 'assets')
        salida_grafica = os.path.join(assets_dir if os.path.exists(assets_dir) else base_proj, 'eda_report.png')
    clases = sorted([d for d in os.listdir(dir_raw) if os.path.isdir(os.path.join(dir_raw, d))])
    
    conteos_raw = {}
    varianzas_blur = []
    dimensiones = []
    
    ejemplo_raw = None
    ejemplo_clase = None
    
    for cls in clases:
        archivos = glob.glob(os.path.join(dir_raw, cls, '*.*'))
        conteos_raw[cls] = len(archivos)
        for arc in archivos:
            img = cv2.imread(arc)
            if img is not None:
                if ejemplo_raw is None:
                    ejemplo_raw = img
                    ejemplo_clase = cls
                meta = auditar_calidad_imagen(img)
                varianzas_blur.append(meta['varianza_laplaciano'])
                dimensiones.append((img.shape[1], img.shape[0]))
                
    # Crear figura de reporte EDA
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('REPORTE DE ANÁLISIS EXPLORATORIO Y CALIDAD DEL DATASET', fontsize=15, fontweight='bold', y=0.98)
    
    # 1. Distribución de Clases
    axes[0, 0].bar(conteos_raw.keys(), conteos_raw.values(), color=['#e67e22', '#2ecc71', '#c0392b'], edgecolor='black')
    axes[0, 0].set_title('1. Distribución de Imágenes por Patología Foliar', fontweight='bold')
    axes[0, 0].set_ylabel('Cantidad de Imágenes')
    axes[0, 0].grid(axis='y', linestyle=':', alpha=0.7)
    for i, (k, v) in enumerate(conteos_raw.items()):
        axes[0, 0].text(i, v + 0.2, str(v), ha='center', fontweight='bold')
        
    # 2. Histograma de Nitidez (Varianza Laplaciana)
    axes[0, 1].hist(varianzas_blur, bins=15, color='#3498db', edgecolor='black', alpha=0.8)
    axes[0, 1].axvline(80.0, color='red', linestyle='--', linewidth=2, label='Umbral Nitidez (80.0)')
    axes[0, 1].set_title('2. Distribución de Nitidez (Varianza del Laplaciano)', fontweight='bold')
    axes[0, 1].set_xlabel('Varianza Laplaciana (Mayor = Más Nítida)')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].legend()
    axes[0, 1].grid(linestyle=':', alpha=0.7)
    
    # 3. Comparativa de Imagen: Original vs Preprocesada
    if ejemplo_raw is not None:
        denoise = reducir_ruido_bilateral(ejemplo_raw)
        clahe = realzar_contraste_clahe_lab(denoise)
        
        comp_img = np.hstack([
            cv2.cvtColor(ejemplo_raw, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(clahe, cv2.COLOR_BGR2RGB)
        ])
        axes[1, 0].imshow(comp_img)
        axes[1, 0].set_title(f'3. Muestra ({ejemplo_clase}): Original (Izq.) vs CLAHE+Filtro (Der.)', fontweight='bold')
        axes[1, 0].axis('off')
        
        # 4. Comparativa de Histogramas de Color (Canal de Luminancia L)
        lab_orig = cv2.cvtColor(ejemplo_raw, cv2.COLOR_BGR2LAB)
        lab_clahe = cv2.cvtColor(clahe, cv2.COLOR_BGR2LAB)
        
        hist_orig = cv2.calcHist([lab_orig], [0], None, [256], [0, 256])
        hist_clahe = cv2.calcHist([lab_clahe], [0], None, [256], [0, 256])
        
        axes[1, 1].plot(hist_orig, color='gray', linestyle='--', label='Luminancia Original')
        axes[1, 1].plot(hist_clahe, color='#8e44ad', linewidth=2, label='Luminancia con CLAHE')
        axes[1, 1].set_title('4. Histograma de Luminancia L* (CIE-Lab)', fontweight='bold')
        axes[1, 1].set_xlabel('Intensidad de Gris (0 - 255)')
        axes[1, 1].set_ylabel('Densidad de Píxeles')
        axes[1, 1].legend()
        axes[1, 1].grid(linestyle=':', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig(salida_grafica, dpi=150)
    plt.close()
    print(f" Reporte gráfico guardado en: {salida_grafica}")


if __name__ == '__main__':
    generar_reporte_eda()
