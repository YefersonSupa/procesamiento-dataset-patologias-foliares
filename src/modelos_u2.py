"""
Definición de Arquitecturas de Redes Convolucionales para Clasificación Foliar
Proyecto Semestral: Comparación de 3 Modelos en Visión Artificial
Asignatura: Visión Computacional (IF657BIN)
Autores:
  - SUPA CUSIPAUCAR, Yeferson (220553)
  - VILLAVICENCIO SEGUIL, Edu Piero (220555)
Docente: CCACYAHUILLCA BEJAR, HANS HARLEY
UNSAAC - 2026-2
"""

import os
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    mobilenet_v2, MobileNet_V2_Weights
)
from typing import Dict, Any, Tuple


class CustomPotatoCNN(nn.Module):
    """
    Arquitectura Convolucional Personalizada diseñada desde cero (Baseline).
    Estructurada en 3 bloques jerárquicos de convolución profunda con Batch Normalization,
    activaciones ReLU, Max Pooling y Dropout regularizador.
    """
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.3):
        super(CustomPotatoCNN, self).__init__()
        
        # Bloque Convolucional 1: Extracción de primitivas (bordes, contrastes)
        self.bloque1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 256 -> 128
        )
        
        # Bloque Convolucional 2: Texturas y patrones foliares intermedios
        self.bloque2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 128 -> 64
        )
        
        # Bloque Convolucional 3: Lesiones patológicas de alto nivel semántico
        self.bloque3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64 -> 32
        )
        
        # Adaptación dimensional global
        self.adapt_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Clasificador denso completamente conectado
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bloque1(x)
        x = self.bloque2(x)
        x = self.bloque3(x)
        x = self.adapt_pool(x)
        x = self.clasificador(x)
        return x


class ResNet18Potato(nn.Module):
    """
    Arquitectura Residual ResNet-18 adaptada mediante Transfer Learning.
    Aprovecha conexiones residuales (skip connections) para mitigar el desvanecimiento
    de gradientes y transferir representaciones robustas aprendidas en ImageNet.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = False):
        super(ResNet18Potato, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Descongela todas las capas para fine-tuning completo."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class MobileNetV2Potato(nn.Module):
    """
    Arquitectura Eficiente MobileNetV2 adaptada mediante Transfer Learning.
    Basada en Convoluciones Separables en Profundidad (Depthwise Separable Convolutions)
    y Bloques Residuales Invertidos con Cuellos de Botella Lineales, óptima para
    despliegue en dispositivos móviles y sistemas embebidos de agricultura de precisión.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = False):
        super(MobileNetV2Potato, self).__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Descongela capas para fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def construir_modelo(
    nombre_modelo: str, 
    num_classes: int = 3, 
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Fábrica de instanciación de modelos según identificador.
    """
    nombre = nombre_modelo.lower()
    if 'custom' in nombre:
        return CustomPotatoCNN(num_classes=num_classes)
    elif 'resnet' in nombre:
        return ResNet18Potato(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif 'mobilenet' in nombre:
        return MobileNetV2Potato(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Modelo desconocido: '{nombre_modelo}'. Opciones: ['custom', 'resnet18', 'mobilenetv2']")


def obtener_resumen_arquitectura(modelo: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 256, 256)) -> Dict[str, Any]:
    """
    Calcula parámetros totales, entrenables y tamaño en disco estimado en MB.
    """
    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    
    # Estimación de memoria en MB (pesos float32 = 4 bytes)
    size_mb = round((total_params * 4) / (1024 * 1024), 2)
    
    return {
        'total_parametros': total_params,
        'parametros_entrenables': trainable_params,
        'tamano_estimado_mb': size_mb,
        'input_size': list(input_size)
    }


if __name__ == '__main__':
    print(" Verificando instanciación y dimensiones de las 3 arquitecturas:")
    tensor_test = torch.randn(2, 3, 256, 256)
    
    for mod_nombre in ['custom', 'resnet18', 'mobilenetv2']:
        m = construir_modelo(mod_nombre)
        m.eval()
        with torch.no_grad():
            salida = m(tensor_test)
        res = obtener_resumen_arquitectura(m)
        print(f"[{mod_nombre.upper()}] Salida: {salida.shape} | Params Totales: {res['total_parametros']:,} | Tamaño: {res['tamano_estimado_mb']} MB")
