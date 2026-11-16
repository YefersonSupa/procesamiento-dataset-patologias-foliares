"""
Suite de Pruebas Unitarias para las Arquitecturas Convolucionales (Unidad 2)
Proyecto Semestral: Comparación de 3 Modelos en Visión Artificial
UNSAAC - EPIIS
Docente: CCACYAHUILLCA BEJAR, HANS HARLEY
Autores:
  - SUPA CUSIPAUCAR, Yeferson - 220553
  - VILLAVICENCIO SEGUIL, Edu Piero - 220555
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from modelos_u2 import (
    CustomPotatoCNN,
    ResNet18Potato,
    MobileNetV2Potato,
    construir_modelo,
    obtener_resumen_arquitectura
)


class TestModelosU2(unittest.TestCase):
    
    def setUp(self):
        self.tensor_test = torch.randn(2, 3, 256, 256)

    def test_custom_potato_cnn_forward(self):
        modelo = CustomPotatoCNN(num_classes=3)
        salida = modelo(self.tensor_test)
        self.assertEqual(salida.shape, (2, 3))
        self.assertFalse(torch.isnan(salida).any().item())

    def test_resnet18_potato_forward(self):
        modelo = ResNet18Potato(num_classes=3, pretrained=False)
        salida = modelo(self.tensor_test)
        self.assertEqual(salida.shape, (2, 3))
        self.assertFalse(torch.isnan(salida).any().item())

    def test_mobilenet_v2_potato_forward(self):
        modelo = MobileNetV2Potato(num_classes=3, pretrained=False)
        salida = modelo(self.tensor_test)
        self.assertEqual(salida.shape, (2, 3))
        self.assertFalse(torch.isnan(salida).any().item())

    def test_fabrica_construir_modelo(self):
        m1 = construir_modelo('custom', pretrained=False)
        m2 = construir_modelo('resnet18', pretrained=False)
        m3 = construir_modelo('mobilenetv2', pretrained=False)
        
        self.assertIsInstance(m1, CustomPotatoCNN)
        self.assertIsInstance(m2, ResNet18Potato)
        self.assertIsInstance(m3, MobileNetV2Potato)
        
        with self.assertRaises(ValueError):
            construir_modelo('modelo_inexistente')

    def test_resumen_arquitectura(self):
        modelo = CustomPotatoCNN(num_classes=3)
        resumen = obtener_resumen_arquitectura(modelo)
        self.assertIn('total_parametros', resumen)
        self.assertIn('parametros_entrenables', resumen)
        self.assertIn('tamano_estimado_mb', resumen)
        self.assertGreater(resumen['total_parametros'], 0)
        self.assertGreater(resumen['tamano_estimado_mb'], 0)


if __name__ == '__main__':
    unittest.main()
