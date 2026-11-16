"""
Módulo para Extensión e Ingesta del Dataset Completo (PLD 3 Classes 256)
Proyecto Semestral: Detección y Clasificación de Patologías Foliares en Papa
Asignatura: Visión Computacional (IF657BIN)
Docente: CCACYAHUILLCA BEJAR, HANS HARLEY
Autores:
  - SUPA CUSIPAUCAR, Yeferson (220553)
  - VILLAVICENCIO SEGUIL, Edu Piero (220555)
UNSAAC - 2026-2
"""

import os
import zipfile
import json
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple


def extraer_dataset_zip(
    ruta_zip: str = None,
    dir_destino: str = None,
    forzar: bool = False
) -> str:
    """
    Extrae el contenido de archive.zip en el directorio data_raw especificado.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ruta_zip is None:
        ruta_zip = os.path.join(base_dir, 'archive.zip')
    if dir_destino is None:
        dir_destino = os.path.join(base_dir, 'data_raw')
        
    pld_dir = os.path.join(dir_destino, 'PLD_3_Classes_256')
    if os.path.exists(pld_dir) and not forzar:
        print(f"[INFO] Dataset ya extraído en: {pld_dir}")
        return pld_dir
        
    print(f"[INFO] Extrayendo {ruta_zip} hacia {dir_destino}...")
    t0 = time.time()
    with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
        zip_ref.extractall(dir_destino)
    print(f"[OK] Extracción completada en {time.time() - t0:.2f} segundos.")
    return pld_dir


def auditar_dataset_extendido(dir_dataset: str) -> Dict[str, Any]:
    """
    Audita el censo de imágenes, resoluciones, canales y métricas de nitidez (Laplaciano)
    a lo largo de las particiones Training, Validation y Testing.
    """
    splits = ['Training', 'Validation', 'Testing']
    clases = ['Early_Blight', 'Healthy', 'Late_Blight']
    
    reporte = {
        'nombre_dataset': 'Potato Leaf Disease (PLD) 3 Classes 256',
        'resolucion_esperada': [256, 256],
        'clases': clases,
        'distribucion_splits': {},
        'metricas_calidad_por_clase': {},
        'resumen_global': {
            'total_imagenes': 0,
            'imagenes_validas': 0,
            'imagenes_nitidas': 0
        }
    }
    
    varianzas_laplaciano_por_clase = {c: [] for c in clases}
    
    for sp in splits:
        reporte['distribucion_splits'][sp] = {}
        for cls in clases:
            dir_clase = os.path.join(dir_dataset, sp, cls)
            if not os.path.exists(dir_clase):
                conteo = 0
            else:
                archivos = [f for f in os.listdir(dir_clase) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                conteo = len(archivos)
                
                # Muestreo de auditoría de calidad (primeros 50 archivos de cada grupo)
                for f in archivos[:50]:
                    ruta_img = os.path.join(dir_clase, f)
                    img = cv2.imread(ruta_img)
                    if img is not None:
                        reporte['resumen_global']['imagenes_validas'] += 1
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        var_lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        varianzas_laplaciano_por_clase[cls].append(var_lap)
                        if var_lap >= 80.0:
                            reporte['resumen_global']['imagenes_nitidas'] += 1
                            
            reporte['distribucion_splits'][sp][cls] = conteo
            reporte['resumen_global']['total_imagenes'] += conteo

    for cls in clases:
        vals = varianzas_laplaciano_por_clase[cls]
        reporte['metricas_calidad_por_clase'][cls] = {
            'varianza_laplaciano_media': round(float(np.mean(vals)), 2) if vals else 0.0,
            'varianza_laplaciano_std': round(float(np.std(vals)), 2) if vals else 0.0,
            'tasa_nitidez_muestreada': round(float(np.mean([v >= 80.0 for v in vals])), 4) if vals else 0.0
        }
        
    return reporte


def ejecutar_extension_dataset() -> Dict[str, Any]:
    """
    Función principal de ejecución de la ingesta y auditoría.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_zip = os.path.join(base_dir, 'archive.zip')
    dir_raw = os.path.join(base_dir, 'data_raw')
    
    pld_dir = extraer_dataset_zip(ruta_zip, dir_raw)
    reporte = auditar_dataset_extendido(pld_dir)
    
    ruta_reporte = os.path.join(base_dir, 'metadata_dataset_u2.json')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
        
    print("=" * 60)
    print(" REPORTE DE EXTENSIÓN Y CENSO DEL DATASET (UNIDAD 2)")
    print("=" * 60)
    print(f" Total de imágenes en dataset extendido: {reporte['resumen_global']['total_imagenes']}")
    for sp, datos in reporte['distribucion_splits'].items():
        subtotal = sum(datos.values())
        print(f" - {sp} ({subtotal} imgs): {datos}")
    print(f" Metadatos serializados en: {ruta_reporte}")
    return reporte


if __name__ == '__main__':
    ejecutar_extension_dataset()
