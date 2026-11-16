"""
Módulo Especializado de Aumento de Datos Foliar (Data Augmentation Engine)
Proyecto Semestral: Detección y Clasificación de Patologías Foliares en Papa
Asignatura: Visión Computacional (IF657BIN)
Docente: CCACYAHUILLCA BEJAR, HANS HARLEY
Autores:
  - SUPA CUSIPAUCAR, Yeferson - 220553
  - VILLAVICENCIO SEGUIL, Edu Piero - 220555
UNSAAC - 2026-2
"""

import os
import cv2
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional


def rotacion_aleatoria(img_bgr: np.ndarray, max_angulo: float = 30.0) -> np.ndarray:
    """Aplica rotación afín con borde reflectivo para evitar franjas negras."""
    h, w = img_bgr.shape[:2]
    angulo = np.random.uniform(-max_angulo, max_angulo)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def volteo_simetrico(img_bgr: np.ndarray, modo: str = 'ambos') -> np.ndarray:
    """Aplica giro horizontal, vertical o aleatorio."""
    if modo == 'horizontal':
        return cv2.flip(img_bgr, 1)
    elif modo == 'vertical':
        return cv2.flip(img_bgr, 0)
    else:
        opciones = [1, 0, -1] # 1=horizontal, 0=vertical, -1=ambos
        elegido = np.random.choice(opciones)
        return cv2.flip(img_bgr, elegido)


def fluctuacion_iluminacion_hsv(
    img_bgr: np.ndarray,
    delta_h: int = 8,
    escala_s: Tuple[float, float] = (0.85, 1.15),
    escala_v: Tuple[float, float] = (0.80, 1.20)
) -> np.ndarray:
    """
    Modifica tono, saturación y brillo en el espacio cromático HSV
    simulando distintas horas de radiación solar y sombras en campo.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Micro-fluctuación de tonalidad (H: 0..179)
    dh = np.random.uniform(-delta_h, delta_h)
    hsv[:, :, 0] = (hsv[:, :, 0] + dh) % 180.0
    
    # Escalamiento de saturación (S: 0..255)
    ds = np.random.uniform(escala_s[0], escala_s[1])
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * ds, 0, 255)
    
    # Escalamiento de valor/brillo (V: 0..255)
    dv = np.random.uniform(escala_v[0], escala_v[1])
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * dv, 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def agregar_ruido_gaussiano(img_bgr: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Simula ruido térmico ISO del sensor de la cámara en capturas agrícolas."""
    ruido = np.random.normal(0, sigma, img_bgr.shape)
    img_ruidosa = img_bgr.astype(np.float32) + ruido
    return np.clip(img_ruidosa, 0, 255).astype(np.uint8)


def micro_desenfoque_movimiento(img_bgr: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Simula desenfoque suave provocado por el viento o vibración del dron."""
    return cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)


def zoom_y_traslacion(
    img_bgr: np.ndarray, 
    escala_min: float = 0.85, 
    max_shift: int = 15
) -> np.ndarray:
    """Aplica recorte con zoom y traslación subpíxel simulando distancia focal variable."""
    h, w = img_bgr.shape[:2]
    factor = np.random.uniform(escala_min, 1.0)
    nuevo_h, nuevo_w = int(h * factor), int(w * factor)
    
    # Punto de recorte
    y1 = np.random.randint(0, h - nuevo_h + 1)
    x1 = np.random.randint(0, w - nuevo_w + 1)
    recorte = img_bgr[y1:y1 + nuevo_h, x1:x1 + nuevo_w]
    
    # Redimensionar al tamaño original
    redim = cv2.resize(recorte, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Traslación leve
    tx = np.random.randint(-max_shift, max_shift + 1)
    ty = np.random.randint(-max_shift, max_shift + 1)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(redim, M_trans, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def cutout_oclusion(
    img_bgr: np.ndarray, 
    num_oclusiones: int = 2, 
    tam_max: int = 30
) -> np.ndarray:
    """Simula oclusiones físicas (gotas de agua, restos de suelo, tallos superpuestos)."""
    resultado = img_bgr.copy()
    h, w = resultado.shape[:2]
    
    for _ in range(num_oclusiones):
        th = np.random.randint(10, tam_max)
        tw = np.random.randint(10, tam_max)
        y = np.random.randint(0, h - th)
        x = np.random.randint(0, w - tw)
        # Relleno con tono neutro o promedio foliar
        color = np.random.choice([0, 30, 80])
        resultado[y:y + th, x:x + tw] = (color, color, color)
        
    return resultado


def aumentar_imagen_pipeline(
    img_bgr: np.ndarray,
    aplicar_rot: bool = True,
    aplicar_flip: bool = True,
    aplicar_hsv: bool = True,
    aplicar_ruido: bool = True,
    aplicar_zoom: bool = True,
    aplicar_cutout: bool = False
) -> np.ndarray:
    """
    Ejecuta un ciclo integrado de aumento probabilístico sobre una imagen foliar.
    """
    out = img_bgr.copy()
    
    if aplicar_rot and np.random.rand() > 0.3:
        out = rotacion_aleatoria(out, max_angulo=25.0)
        
    if aplicar_flip and np.random.rand() > 0.4:
        out = volteo_simetrico(out, modo='ambos')
        
    if aplicar_hsv and np.random.rand() > 0.3:
        out = fluctuacion_iluminacion_hsv(out)
        
    if aplicar_zoom and np.random.rand() > 0.5:
        out = zoom_y_traslacion(out)
        
    if aplicar_ruido and np.random.rand() > 0.6:
        out = agregar_ruido_gaussiano(out, sigma=8.0)
        
    if aplicar_cutout and np.random.rand() > 0.7:
        out = cutout_oclusion(out, num_oclusiones=2, tam_max=20)
        
    return out


def generar_mosaico_comparativo(ruta_img_origen: str, ruta_salida_figura: str):
    """
    Toma una imagen y genera un mosaico de 9 paneles demostrando cada técnica de aumento.
    """
    img = cv2.imread(ruta_img_origen)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_img_origen}")
        
    # Aplicar técnicas individuales de forma determinista para la demostración
    np.random.seed(42)
    tecnicas = [
        ("Original (Sin Aumento)", img),
        ("1. Rotación Afín (25°)", rotacion_aleatoria(img, 25.0)),
        ("2. Volteo Horizontal", cv2.flip(img, 1)),
        ("3. Volteo Vertical", cv2.flip(img, 0)),
        ("4. Fluctuación HSV (Luz)", fluctuacion_iluminacion_hsv(img, delta_h=10, escala_v=(1.2, 1.2))),
        ("5. Zoom y Traslación", zoom_y_traslacion(img, escala_min=0.85, max_shift=12)),
        ("6. Ruido Gaussiano (Sensor)", agregar_ruido_gaussiano(img, sigma=18.0)),
        ("7. Desenfoque por Viento", micro_desenfoque_movimiento(img, ksize=5)),
        ("8. Pipeline Combinado", aumentar_imagen_pipeline(img, aplicar_cutout=True))
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(11, 11), dpi=300)
    for idx, (titulo, im_bgr) in enumerate(tecnicas):
        r, c = idx // 3, idx % 3
        ax = axes[r, c]
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(im_rgb)
        ax.set_title(titulo, fontsize=10, fontweight='bold', pad=6)
        ax.axis('off')
        
    plt.suptitle("Técnicas de Aumento de Datos Foliar (Data Augmentation Engine)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(ruta_salida_figura), exist_ok=True)
    plt.savefig(ruta_salida_figura, bbox_inches='tight')
    plt.close()
    print(f"[OK] Mosaico de aumento guardado en: {ruta_salida_figura}")


def aumentar_dataset_directorio(
    dir_origen: str,
    dir_destino: str,
    factor_aumento: int = 2,
    max_imagenes_por_clase: Optional[int] = None
) -> Dict[str, int]:
    """
    Recorre un directorio con carpetas por clases y genera N imágenes aumentadas por cada imagen original.
    """
    clases = [d for d in os.listdir(dir_origen) if os.path.isdir(os.path.join(dir_origen, d)) and not d.startswith(('.', '_'))]
    resumen = {}
    
    for cls in clases:
        ruta_clase_in = os.path.join(dir_origen, cls)
        ruta_clase_out = os.path.join(dir_destino, cls)
        os.makedirs(ruta_clase_out, exist_ok=True)
        
        archivos = glob.glob(os.path.join(ruta_clase_in, '*.[jJ][pP][gG]')) + \
                   glob.glob(os.path.join(ruta_clase_in, '*.[pP][nN][gG]'))
                   
        if max_imagenes_por_clase:
            archivos = archivos[:max_imagenes_por_clase]
            
        conteo_generado = 0
        for ruta_img in archivos:
            base_name = os.path.splitext(os.path.basename(ruta_img))[0]
            img = cv2.imread(ruta_img)
            if img is None:
                continue
                
            # Copiar original a destino
            cv2.imwrite(os.path.join(ruta_clase_out, f"{base_name}_orig.png"), img)
            conteo_generado += 1
            
            # Generar N aumentos
            for k in range(factor_aumento):
                img_aug = aumentar_imagen_pipeline(img)
                salida_aug = os.path.join(ruta_clase_out, f"{base_name}_aug{k+1}.png")
                cv2.imwrite(salida_aug, img_aug)
                conteo_generado += 1
                
        resumen[cls] = conteo_generado
        print(f"[AUMENTO] Clase '{cls}': {len(archivos)} originales -> {conteo_generado} totales (Factor x{factor_aumento+1})")
        
    return resumen


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Buscar una imagen de muestra para generar el mosaico comparativo
    muestras = glob.glob(os.path.join(base_dir, 'data_raw', 'Early_Blight', '*.JPG')) + \
               glob.glob(os.path.join(base_dir, 'data_raw', 'Early_Blight', '*.jpg'))
               
    if muestras:
        ruta_sample = muestras[0]
        ruta_figura = os.path.join(base_dir, 'assets', 'figuras_modelos', 'demostracion_data_augmentation.png')
        generar_mosaico_comparativo(ruta_sample, ruta_figura)
