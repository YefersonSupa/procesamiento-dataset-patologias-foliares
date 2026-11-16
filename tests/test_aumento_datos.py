"""
Pruebas Unitarias para el Módulo de Aumento de Datos Foliar
UNSAAC - Visión Computacional
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from aumento_datos import (
    rotacion_aleatoria,
    volteo_simetrico,
    fluctuacion_iluminacion_hsv,
    agregar_ruido_gaussiano,
    micro_desenfoque_movimiento,
    zoom_y_traslacion,
    cutout_oclusion,
    aumentar_imagen_pipeline
)


class TestAumentoDatos(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.img_test = np.ones((256, 256, 3), dtype=np.uint8) * 128

    def test_rotacion_aleatoria(self):
        out = rotacion_aleatoria(self.img_test, max_angulo=15.0)
        self.assertEqual(out.shape, self.img_test.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_volteo_simetrico(self):
        out = volteo_simetrico(self.img_test, modo='horizontal')
        self.assertEqual(out.shape, self.img_test.shape)

    def test_fluctuacion_hsv(self):
        out = fluctuacion_iluminacion_hsv(self.img_test)
        self.assertEqual(out.shape, self.img_test.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_ruido_gaussiano(self):
        out = agregar_ruido_gaussiano(self.img_test, sigma=10.0)
        self.assertEqual(out.shape, self.img_test.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_micro_desenfoque(self):
        out = micro_desenfoque_movimiento(self.img_test, ksize=3)
        self.assertEqual(out.shape, self.img_test.shape)

    def test_zoom_y_traslacion(self):
        out = zoom_y_traslacion(self.img_test)
        self.assertEqual(out.shape, self.img_test.shape)

    def test_cutout(self):
        out = cutout_oclusion(self.img_test, num_oclusiones=2, tam_max=20)
        self.assertEqual(out.shape, self.img_test.shape)

    def test_pipeline_completo(self):
        out = aumentar_imagen_pipeline(self.img_test)
        self.assertEqual(out.shape, (256, 256, 3))
        self.assertEqual(out.dtype, np.uint8)


if __name__ == '__main__':
    unittest.main()
