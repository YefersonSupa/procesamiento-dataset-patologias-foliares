"""
Generador de infografía comparativa del pipeline de preprocesamiento.
Genera la figura 'pipeline_comparativa.png' para el informe y diapositivas.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from preprocesamiento import (
        reducir_ruido_bilateral,
        realzar_contraste_clahe_lab,
        redimensionar_letterbox,
        aplicar_data_augmentation,
        auditar_calidad_imagen
    )
except ImportError:
    from src.preprocesamiento import (
        reducir_ruido_bilateral,
        realzar_contraste_clahe_lab,
        redimensionar_letterbox,
        aplicar_data_augmentation,
        auditar_calidad_imagen
    )


def generar_comparativa():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data_raw', 'Early_Blight')
    
    archivos = glob.glob(os.path.join(raw_dir, '*.*'))
    if not archivos:
        print("No se encontraron imagenes en", raw_dir)
        return
        
    img_path = archivos[0]
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("No se pudo leer:", img_path)
        return
        
    meta = auditar_calidad_imagen(img_bgr)
    
    # Etapas
    paso1_bilateral = reducir_ruido_bilateral(img_bgr, d=9, sigma_color=75.0, sigma_space=75.0)
    paso2_clahe = realzar_contraste_clahe_lab(paso1_bilateral, clip_limit=2.5, tile_grid_size=(8, 8))
    paso3_letterbox = redimensionar_letterbox(paso2_clahe, target_size=(256, 256))
    
    np.random.seed(42)
    paso4_augmented = aplicar_data_augmentation(paso3_letterbox, prob_flip=1.0, max_angulo_rotacion=15.0)
    
    # Convertir a RGB para matplotlib
    img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_bilateral = cv2.cvtColor(paso1_bilateral, cv2.COLOR_BGR2RGB)
    img_rgb_clahe = cv2.cvtColor(paso2_clahe, cv2.COLOR_BGR2RGB)
    img_rgb_letterbox = cv2.cvtColor(paso3_letterbox, cv2.COLOR_BGR2RGB)
    img_rgb_augmented = cv2.cvtColor(paso4_augmented, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    fig.patch.set_facecolor('#ffffff')
    
    titulos = [
        f"1. Original (In-the-wild)\n{img_bgr.shape[1]}x{img_bgr.shape[0]} | Var Lap: {meta['varianza_laplaciano']}",
        "2. Filtro Bilateral\nPreservación de bordes (d=9)",
        "3. CLAHE en CIE-Lab\nRealce local sin saturar color",
        "4. Letterboxing 256x256\nAspect Ratio 1:1 con padding",
        "5. Data Augmentation\nRotación ±15° y Flip (Train)"
    ]
    
    imagenes = [
        img_rgb_orig,
        img_rgb_bilateral,
        img_rgb_clahe,
        img_rgb_letterbox,
        img_rgb_augmented
    ]
    
    colores_borde = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    for ax, img, titulo, color in zip(axes, imagenes, titulos, colores_borde):
        ax.imshow(img)
        ax.set_title(titulo, fontsize=10.5, fontweight='bold', pad=8, color='#1e293b')
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
            
    assets_dir = os.path.join(base_dir, 'assets')
    out_path = os.path.join(assets_dir if os.path.exists(assets_dir) else base_dir, 'pipeline_comparativa.png')
    plt.savefig(out_path, dpi=250, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("Figura guardada con exito en:", out_path)

if __name__ == '__main__':
    generar_comparativa()
