"""
Módulo de Preprocesamiento de Imágenes para Visión Computacional
Proyecto Semestral: Detección y Clasificación de Patologías Foliares en Papa
Autor: SUPA CUSIPAUCAR, Yeferson - 220553
Docente: CCACYAHUILLCA BEJAR, HANS HARLEY
UNSAAC - 2026-2
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional


def auditar_calidad_imagen(img_bgr: np.ndarray, umbral_blur: float = 80.0) -> Dict[str, Any]:
    """
    Audita la calidad de la imagen calculando la nitidez mediante la varianza del operador Laplaciano.
    Permite descartar capturas con desenfoque de movimiento o fuera de foco.
    
    Args:
        img_bgr: Imagen en formato BGR.
        umbral_blur: Varianza mínima del Laplaciano para considerar la imagen nítida.
        
    Returns:
        Diccionario con métricas de calidad y bandera booleana 'es_nitida'.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {'es_valida': False, 'es_nitida': False, 'varianza_laplaciano': 0.0}
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    var_laplaciano = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    es_nitida = var_laplaciano >= umbral_blur
    
    return {
        'es_valida': True,
        'es_nitida': es_nitida,
        'varianza_laplaciano': round(var_laplaciano, 2),
        'resolucion_original': (img_bgr.shape[1], img_bgr.shape[0])
    }


def redimensionar_letterbox(
    img_bgr: np.ndarray, 
    target_size: Tuple[int, int] = (256, 256), 
    color_fondo: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Redimensiona la imagen al tamaño objetivo manteniendo la relación de aspecto original (Aspect Ratio)
    mediante letterboxing (relleno uniforme de bordes), evitando distorsiones morfológicas en la hoja.
    
    Args:
        img_bgr: Imagen de entrada.
        target_size: Tupla (ancho, alto) de salida.
        color_fondo: Color en formato BGR para los bordes de relleno (por defecto negro).
        
    Returns:
        Imagen redimensionada con padding.
    """
    target_w, target_h = target_size
    h, w = img_bgr.shape[:2]
    
    escala = min(target_w / w, target_h / h)
    nuevo_w = int(w * escala)
    nuevo_h = int(h * escala)
    
    interpolacion = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_CUBIC
    redimensionada = cv2.resize(img_bgr, (nuevo_w, nuevo_h), interpolation=interpolacion)
    
    # Crear lienzo con color de fondo
    lienzo = np.full((target_h, target_w, 3), color_fondo, dtype=np.uint8)
    
    # Centrar la imagen en el lienzo
    pad_top = (target_h - nuevo_h) // 2
    pad_left = (target_w - nuevo_w) // 2
    lienzo[pad_top:pad_top + nuevo_h, pad_left:pad_left + nuevo_w] = redimensionada
    
    return lienzo


def reducir_ruido_bilateral(
    img_bgr: np.ndarray, 
    d: int = 9, 
    sigma_color: float = 75.0, 
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Aplica un filtro bilateral espacial para atenuar ruido digital del sensor fotográfico
    conservando nítidos los gradientes y bordes de las lesiones patológicas.
    
    Args:
        img_bgr: Imagen BGR.
        d: Diámetro del vecindario de píxeles.
        sigma_color: Desviación estándar en el espacio de color.
        sigma_space: Desviación estándar en el espacio de coordenadas.
        
    Returns:
        Imagen filtrada sin pérdida de bordes.
    """
    return cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def segmentar_hoja_hsv(
    img_bgr: np.ndarray, 
    kernel_morfologia: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmenta la masa foliar vegetal eliminando el fondo mediante umbralización en el espacio HSV
    y refinamiento morfológico (Apertura para eliminar ruido y Cierre para sellar huecos foliares).
    
    Args:
        img_bgr: Imagen BGR.
        kernel_morfologia: Tamaño del elemento estructurante elíptico.
        
    Returns:
        Tupla (imagen_enmascarada_bgr, mascara_binaria).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Rango para tonalidades verdes y clorofila foliar (incluyendo tonos amarillentos de clorosis)
    verde_bajo = np.array([20, 35, 30], dtype=np.uint8)
    verde_alto = np.array([90, 255, 255], dtype=np.uint8)
    
    mascara_verde = cv2.inRange(hsv, verde_bajo, verde_alto)
    
    # Rango para lesiones necróticas marrones/oscuras típicas de tizón
    marron_bajo = np.array([5, 40, 20], dtype=np.uint8)
    marron_alto = np.array([25, 255, 180], dtype=np.uint8)
    mascara_marron = cv2.inRange(hsv, marron_bajo, marron_alto)
    
    # Unión de ambas máscaras
    mascara_total = cv2.bitwise_or(mascara_verde, mascara_marron)
    
    # Refinamiento morfológico con elemento estructurante elíptico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_morfologia, kernel_morfologia))
    mascara_limpia = cv2.morphologyEx(mascara_total, cv2.MORPH_OPEN, kernel, iterations=2)
    mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Si la máscara captura suficiente área, se aplica; de lo contrario, se conserva la imagen original
    if np.sum(mascara_limpia > 0) > (0.15 * mascara_limpia.size):
        img_segmentada = cv2.bitwise_and(img_bgr, img_bgr, mask=mascara_limpia)
    else:
        img_segmentada = img_bgr.copy()
        mascara_limpia = np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255
        
    return img_segmentada, mascara_limpia


def realzar_contraste_clahe_lab(
    img_bgr: np.ndarray, 
    clip_limit: float = 2.5, 
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Aplica ecualización adaptativa de histograma limitada por contraste (CLAHE) en el canal de
    Luminancia (L) del espacio CIE-Lab, evitando sobresaturar los matices de color naturales.
    
    Args:
        img_bgr: Imagen BGR.
        clip_limit: Límite de amplificación de contraste para prevenir ruido excesivo.
        tile_grid_size: Tamaño de las teselas contextuales locales.
        
    Returns:
        Imagen BGR con contraste y micro-detalles realzados.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_canal, a_canal, b_canal = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_canal)
    
    lab_realzado = cv2.merge((l_clahe, a_canal, b_canal))
    return cv2.cvtColor(lab_realzado, cv2.COLOR_LAB2BGR)


def aplicar_data_augmentation(
    img_bgr: np.ndarray, 
    prob_flip: float = 0.5, 
    max_angulo_rotacion: float = 20.0
) -> np.ndarray:
    """
    Genera variaciones controladas de la imagen para enriquecer la variabilidad del dataset.
    Aplica rotación aleatoria leve, espejado y ajustes de brillo/contraste.
    
    Args:
        img_bgr: Imagen BGR procesada.
        prob_flip: Probabilidad de volteo horizontal.
        max_angulo_rotacion: Ángulo máximo de rotación.
        
    Returns:
        Imagen aumentada.
    """
    h, w = img_bgr.shape[:2]
    resultado = img_bgr.copy()
    
    # 1. Flip horizontal aleatorio
    if np.random.rand() < prob_flip:
        resultado = cv2.flip(resultado, 1)
        
    # 2. Rotación leve
    angulo = np.random.uniform(-max_angulo_rotacion, max_angulo_rotacion)
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angulo, 1.0)
    resultado = cv2.warpAffine(resultado, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    # 3. Variación de brillo y contraste suave
    alpha = np.random.uniform(0.9, 1.1) # Contraste
    beta = np.random.uniform(-10, 10)   # Brillo
    resultado = np.clip(alpha * resultado + beta, 0, 255).astype(np.uint8)
    
    return resultado


def pipeline_completo_preprocesamiento(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    aplicar_clahe: bool = True,
    aplicar_denoise: bool = True,
    segmentar_fondo: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ejecuta el pipeline completo de preprocesamiento secuencial sobre una imagen individual.
    
    Secuencia:
    1. Auditoría de nitidez (Laplaciano)
    2. Reducción de ruido bilateral
    3. Segmentación / Filtrado de fondo (opcional)
    4. Realce adaptativo de contraste (CLAHE en CIE-Lab)
    5. Redimensionamiento con Letterbox (conserva aspect ratio)
    """
    metadata = auditar_calidad_imagen(img_bgr)
    if not metadata['es_valida']:
        raise ValueError("Imagen no válida o vacía.")
        
    img_actual = img_bgr.copy()
    
    # 1. Denoising
    if aplicar_denoise:
        img_actual = reducir_ruido_bilateral(img_actual)
        
    # 2. Segmentación de fondo
    if segmentar_fondo:
        img_actual, mascara = segmentar_hoja_hsv(img_actual)
        metadata['area_cobertura_mascara'] = float(np.sum(mascara > 0) / mascara.size)
        
    # 3. Realce de contraste
    if aplicar_clahe:
        img_actual = realzar_contraste_clahe_lab(img_actual)
        
    # 4. Redimensionamiento espacial
    img_salida = redimensionar_letterbox(img_actual, target_size=target_size)
    
    return img_salida, metadata
