"""
Pipeline de Procesamiento por Lotes y Particionado del Dataset
Proyecto Semestral: Preprocesamiento de Dataset de Imágenes
Docente: CCACYAHUILLCA BEJAR, HANS HARLEY
Autor: SUPA CUSIPAUCAR, Yeferson - 220553
UNSAAC - 2026-2
"""

import os
import glob
import json
import shutil
import cv2
import numpy as np
from typing import Dict, List, Tuple
try:
    from preprocesamiento import pipeline_completo_preprocesamiento, aplicar_data_augmentation
except ImportError:
    from src.preprocesamiento import pipeline_completo_preprocesamiento, aplicar_data_augmentation


def procesar_y_particionar_dataset(
    dir_origen: str = None,
    dir_destino: str = None,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    target_size: Tuple[int, int] = (256, 256),
    aumentar_datos_train: bool = True,
    semilla: int = 42
) -> Dict[str, any]:
    """
    Recorre el dataset crudo, ejecuta el preprocesamiento integral en cada imagen
    y organiza el resultado en particiones estratificadas train / val / test.
    """
    base_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dir_origen is None:
        dir_origen = os.path.join(base_proj, 'data_raw')
    if dir_destino is None:
        dir_destino = os.path.join(base_proj, 'data_processed')
    np.random.seed(semilla)
    
    clases = [d for d in os.listdir(dir_origen) if os.path.isdir(os.path.join(dir_origen, d))]
    clases.sort()
    
    # Limpiar o crear directorios destino
    if os.path.exists(dir_destino):
        shutil.rmtree(dir_destino)
        
    for split in ['train', 'val', 'test']:
        for cls in clases:
            os.makedirs(os.path.join(dir_destino, split, cls), exist_ok=True)
            
    resumen_estadistico = {
        'clases': clases,
        'conteo_total_inicial': 0,
        'conteo_total_procesado': 0,
        'distribucion_splits': {'train': 0, 'val': 0, 'test': 0},
        'detalles_por_clase': {}
    }
    
    for cls in clases:
        ruta_clase = os.path.join(dir_origen, cls)
        extensiones = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        archivos = []
        for ext in extensiones:
            archivos.extend(glob.glob(os.path.join(ruta_clase, ext)))
            
        archivos.sort()
        np.random.shuffle(archivos)
        
        n_total = len(archivos)
        resumen_estadistico['conteo_total_inicial'] += n_total
        
        # Calcular índices de partición
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        # Asegurar al menos 1 muestra en cada conjunto si hay suficientes datos
        if n_total >= 3 and n_train == 0:
            n_train = 1
        if n_total >= 3 and n_val == 0:
            n_val = 1
        n_test = n_total - (n_train + n_val)
        
        splits_archivos = {
            'train': archivos[:n_train],
            'val': archivos[n_train:n_train + n_val],
            'test': archivos[n_train + n_val:]
        }
        
        resumen_estadistico['detalles_por_clase'][cls] = {
            'total_raw': n_total,
            'train': len(splits_archivos['train']),
            'val': len(splits_archivos['val']),
            'test': len(splits_archivos['test'])
        }
        
        # Procesar y guardar imágenes
        for split_name, lista_imgs in splits_archivos.items():
            for idx, ruta_img in enumerate(lista_imgs):
                img_bgr = cv2.imread(ruta_img)
                if img_bgr is None:
                    continue
                    
                img_proc, meta = pipeline_completo_preprocesamiento(
                    img_bgr,
                    target_size=target_size,
                    aplicar_clahe=True,
                    aplicar_denoise=True,
                    segmentar_fondo=False
                )
                
                nombre_base = f"{cls}_{idx:04d}.png"
                ruta_salida = os.path.join(dir_destino, split_name, cls, nombre_base)
                cv2.imwrite(ruta_salida, img_proc)
                resumen_estadistico['distribucion_splits'][split_name] += 1
                resumen_estadistico['conteo_total_procesado'] += 1
                
                # En train, generar 1 muestra con data augmentation
                if split_name == 'train' and aumentar_datos_train:
                    img_aug = aplicar_data_augmentation(img_proc)
                    nombre_aug = f"{cls}_{idx:04d}_aug.png"
                    ruta_aug = os.path.join(dir_destino, split_name, cls, nombre_aug)
                    cv2.imwrite(ruta_aug, img_aug)
                    resumen_estadistico['distribucion_splits'][split_name] += 1
                    resumen_estadistico['conteo_total_procesado'] += 1

    # Guardar reporte de metadatos en JSON
    ruta_json = os.path.join(dir_destino, 'metadata_preprocesamiento.json')
    with open(ruta_json, 'w', encoding='utf-8') as f:
        json.dump(resumen_estadistico, f, indent=2, ensure_ascii=False)
        
    print("=" * 60)
    print(" PIPELINE DE PREPROCESAMIENTO FINALIZADO EXITOSAMENTE")
    print("=" * 60)
    print(f" Clases procesadas: {', '.join(clases)}")
    print(f" Total imágenes crudas: {resumen_estadistico['conteo_total_inicial']}")
    print(f" Total procesadas + aumentadas: {resumen_estadistico['conteo_total_procesado']}")
    print(f" Distribución: Train={resumen_estadistico['distribucion_splits']['train']}, "
          f"Val={resumen_estadistico['distribucion_splits']['val']}, "
          f"Test={resumen_estadistico['distribucion_splits']['test']}")
    print(f" Metadatos guardados en: {ruta_json}")
    
    return resumen_estadistico


if __name__ == '__main__':
    procesar_y_particionar_dataset()
