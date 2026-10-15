"""
Suite de Pruebas Unitarias para el Módulo de Preprocesamiento de Imágenes
UNSAAC - Visión Computacional
Autor: SUPA CUSIPAUCAR, Yeferson - 220553
"""

import unittest
import numpy as np
import cv2
import sys
import os

# Asegurar importación desde src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocesamiento import (
    auditar_calidad_imagen,
    redimensionar_letterbox,
    reducir_ruido_bilateral,
    segmentar_hoja_hsv,
    realzar_contraste_clahe_lab,
    aplicar_data_augmentation,
    pipeline_completo_preprocesamiento
)


class TestPreprocesamiento(unittest.TestCase):
    
    def setUp(self):
        # Generar imagen sintética de prueba tipo hoja foliar (200x300 BGR)
        np.random.seed(42)
        self.img_test = np.zeros((200, 300, 3), dtype=np.uint8)
        # Fondo marrón claro
        self.img_test[:, :] = [180, 200, 210]
        # Dibujar una 'hoja' elíptica verde
        cv2.ellipse(self.img_test, (150, 100), (90, 50), 30, 0, 360, (35, 160, 45), -1)
        # Dibujar una lesión necrótica marrón en el centro
        cv2.circle(self.img_test, (150, 100), 18, (20, 60, 110), -1)

    def test_auditar_calidad_valida(self):
        meta = auditar_calidad_imagen(self.img_test, umbral_blur=50.0)
        self.assertTrue(meta['es_valida'])
        self.assertIn('varianza_laplaciano', meta)
        self.assertIn('es_nitida', meta)
        self.assertEqual(meta['resolucion_original'], (300, 200))

    def test_auditar_calidad_invalida(self):
        img_vacia = np.zeros((0, 0, 3), dtype=np.uint8)
        meta = auditar_calidad_imagen(img_vacia)
        self.assertFalse(meta['es_valida'])
        self.assertFalse(meta['es_nitida'])
        
        meta_none = auditar_calidad_imagen(None)
        self.assertFalse(meta_none['es_valida'])

    def test_redimensionar_letterbox_cuadrado(self):
        target = (256, 256)
        res = redimensionar_letterbox(self.img_test, target_size=target)
        self.assertEqual(res.shape, (256, 256, 3))
        self.assertEqual(res.dtype, np.uint8)

    def test_redimensionar_letterbox_relacion_aspecto(self):
        # Probar con entrada no cuadrada y target no cuadrado
        target = (128, 64)
        res = redimensionar_letterbox(self.img_test, target_size=target)
        self.assertEqual(res.shape, (64, 128, 3))

    def test_reducir_ruido_bilateral(self):
        res = reducir_ruido_bilateral(self.img_test, d=5, sigma_color=50, sigma_space=50)
        self.assertEqual(res.shape, self.img_test.shape)
        self.assertEqual(res.dtype, np.uint8)

    def test_realzar_contraste_clahe_lab(self):
        res = realzar_contraste_clahe_lab(self.img_test, clip_limit=2.0)
        self.assertEqual(res.shape, self.img_test.shape)
        self.assertEqual(res.dtype, np.uint8)

    def test_segmentar_hoja_hsv(self):
        seg, mask = segmentar_hoja_hsv(self.img_test)
        self.assertEqual(seg.shape, self.img_test.shape)
        self.assertEqual(mask.shape, self.img_test.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)

    def test_aplicar_data_augmentation(self):
        np.random.seed(123)
        res = aplicar_data_augmentation(self.img_test, prob_flip=1.0, max_angulo_rotacion=10.0)
        self.assertEqual(res.shape, self.img_test.shape)
        self.assertEqual(res.dtype, np.uint8)

    def test_pipeline_completo_preprocesamiento(self):
        img_proc, meta = pipeline_completo_preprocesamiento(
            self.img_test,
            target_size=(256, 256),
            aplicar_clahe=True,
            aplicar_denoise=True,
            segmentar_fondo=False
        )
        self.assertEqual(img_proc.shape, (256, 256, 3))
        self.assertEqual(img_proc.dtype, np.uint8)
        self.assertTrue(meta['es_valida'])


if __name__ == '__main__':
    unittest.main()
