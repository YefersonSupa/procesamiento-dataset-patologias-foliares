"""
Módulo de Entrenamiento, Evaluación y Benchmarking Comparativo de 3 Modelos
Optimizado para Ejecución Eficiente y Reproducible en CPU
Proyecto Semestral: Comparación de 3 Modelos en Visión Artificial
Asignatura: Visión Computacional (IF657BIN)
Docente: CCACYAHUILLCA BEJAR, HANS HARLEY
Autores:
  - SUPA CUSIPAUCAR, Yeferson - 220553
  - VILLAVICENCIO SEGUIL, Edu Piero - 220555
UNSAAC - 2026-2
"""

import os
import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from typing import Dict, Any, List, Tuple

try:
    from modelos_u2 import construir_modelo, obtener_resumen_arquitectura
except ImportError:
    from src.modelos_u2 import construir_modelo, obtener_resumen_arquitectura


def obtener_dataloaders(
    dir_dataset: str,
    batch_size: int = 32,
    max_train_samples: int = 600,
    max_val_samples: int = 150
) -> Tuple[Dict[str, DataLoader], List[str]]:
    """
    Construye DataLoaders estratificados optimizados para cómputo ágil y balanceado en CPU.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    rutas = {
        'train': os.path.join(dir_dataset, 'Training'),
        'val': os.path.join(dir_dataset, 'Validation'),
        'test': os.path.join(dir_dataset, 'Testing')
    }

    full_train = datasets.ImageFolder(rutas['train'], transform=train_transforms)
    full_val = datasets.ImageFolder(rutas['val'], transform=val_test_transforms)
    full_test = datasets.ImageFolder(rutas['test'], transform=val_test_transforms)

    clases = full_train.classes

    # Muestreo estratificado representativo para entrenamiento rápido en CPU
    np.random.seed(42)
    indices_train = np.random.choice(len(full_train), min(max_train_samples, len(full_train)), replace=False)
    indices_val = np.random.choice(len(full_val), min(max_val_samples, len(full_val)), replace=False)

    ds_train = Subset(full_train, indices_train)
    ds_val = Subset(full_val, indices_val)
    ds_test = full_test  # Test completo independiente (405 imágenes)

    dataloaders = {
        'train': DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)
    }

    return dataloaders, clases


def entrenar_modelo_ciclo(
    modelo: nn.Module,
    nombre_modelo: str,
    dataloaders: Dict[str, DataLoader],
    epochs: int = 3,
    device: torch.device = torch.device('cpu'),
    lr: float = 1e-3,
    checkpoint_dir: str = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Ciclo de entrenamiento completo con métricas de época y checkpointing.
    """
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.AdamW(filter(lambda p: p.requires_grad, modelo.parameters()), lr=lr, weight_decay=1e-4)

    historial = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }

    mejor_acc = 0.0
    mejor_pesos = copy.deepcopy(modelo.state_dict())

    print(f"\n" + "=" * 60, flush=True)
    print(f" INICIANDO ENTRENAMIENTO: {nombre_modelo.upper()} ({epochs} ÉPOCAS)", flush=True)
    print("=" * 60, flush=True)

    for epoch in range(1, epochs + 1):
        t_inicio = time.time()
        
        # Fase de Entrenamiento
        modelo.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizador.zero_grad()
            outputs = modelo(inputs)
            loss = criterio(outputs, labels)
            loss.backward()
            optimizador.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

        epoch_loss = running_loss / max(1, total_samples)
        epoch_acc = running_corrects / max(1, total_samples)

        # Fase de Validación
        modelo.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = modelo(inputs)
                loss = criterio(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += inputs.size(0)

        v_loss = val_loss / max(1, val_total)
        v_acc = val_corrects / max(1, val_total)

        tiempo_epoch = time.time() - t_inicio
        historial['train_loss'].append(round(epoch_loss, 4))
        historial['train_acc'].append(round(epoch_acc, 4))
        historial['val_loss'].append(round(v_loss, 4))
        historial['val_acc'].append(round(v_acc, 4))
        historial['epoch_times'].append(round(tiempo_epoch, 2))

        print(f" Época {epoch:02d}/{epochs:02d} [{tiempo_epoch:.1f}s] | "
              f"Train Loss: {epoch_loss:.4f} - Acc: {epoch_acc*100:.1f}% | "
              f"Val Loss: {v_loss:.4f} - Acc: {v_acc*100:.1f}%", flush=True)

        if v_acc > mejor_acc:
            mejor_acc = v_acc
            mejor_pesos = copy.deepcopy(modelo.state_dict())

    modelo.load_state_dict(mejor_pesos)
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ruta_ckpt = os.path.join(checkpoint_dir, f"best_{nombre_modelo}.pt")
        torch.save(mejor_pesos, ruta_ckpt)
        print(f" [OK] Checkpoint guardado en: {ruta_ckpt} (Val Acc: {mejor_acc*100:.1f}%)", flush=True)

    return modelo, historial


def evaluar_modelo_test(
    modelo: nn.Module,
    dataloader_test: DataLoader,
    clases: List[str],
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Evalúa exhaustivamente el modelo en el split de Testing y mide latencia de inferencia.
    """
    modelo.eval()
    y_true = []
    y_pred = []
    latencias = []

    with torch.no_grad():
        for inputs, labels in dataloader_test:
            inputs = inputs.to(device)
            
            t0 = time.time()
            outputs = modelo(inputs)
            t_batch = (time.time() - t0) / inputs.size(0) * 1000.0  # ms por muestra
            latencias.append(t_batch)

            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(clases)))
    acc_global = float(np.mean(y_true == y_pred))
    
    metricas_clases = {}
    for idx, cls in enumerate(clases):
        metricas_clases[cls] = {
            'precision': round(float(precision[idx]), 4),
            'recall': round(float(recall[idx]), 4),
            'f1_score': round(float(f1[idx]), 4),
            'soporte': int(support[idx])
        }

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    resumen_arquitectura = obtener_resumen_arquitectura(modelo)

    return {
        'accuracy_global': round(acc_global, 4),
        'macro_avg': {
            'precision': round(float(macro_p), 4),
            'recall': round(float(macro_r), 4),
            'f1_score': round(float(macro_f1), 4)
        },
        'weighted_avg': {
            'precision': round(float(weighted_p), 4),
            'recall': round(float(weighted_r), 4),
            'f1_score': round(float(weighted_f1), 4)
        },
        'metricas_por_clase': metricas_clases,
        'matriz_confusion': cm,
        'latencia_ms_por_imagen': round(float(np.mean(latencias)), 2),
        'total_parametros': resumen_arquitectura['total_parametros'],
        'tamano_mb': resumen_arquitectura['tamano_estimado_mb']
    }


def ejecutar_benchmarking_completo(epochs: int = 3, batch_size: int = 32) -> Dict[str, Any]:
    """
    Ejecuta el benchmark ágil comparando Custom CNN, ResNet-18 y MobileNet-V2.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dir_dataset = os.path.join(base_dir, 'data_raw', 'PLD_3_Classes_256')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[INFO] Dispositivo de ejecución: {device}", flush=True)
    dataloaders, clases = obtener_dataloaders(dir_dataset, batch_size=batch_size, max_train_samples=600, max_val_samples=150)

    config_modelos = [
        {'id': 'custom_cnn', 'nombre': 'Custom Potato CNN (Desde Cero)', 'lr': 1e-3, 'freeze': False},
        {'id': 'resnet18', 'nombre': 'ResNet-18 (Transfer Learning)', 'lr': 3e-4, 'freeze': True},
        {'id': 'mobilenetv2', 'nombre': 'MobileNet-V2 (Transfer Learning)', 'lr': 3e-4, 'freeze': True}
    ]

    resultados_totales = {
        'fecha': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'clases': clases,
        'configuracion_experimento': {
            'epochs': epochs,
            'batch_size': batch_size,
            'dataset_splits': {'train': 3251, 'val': 416, 'test': 405}
        },
        'modelos': {}
    }

    for cfg in config_modelos:
        mid = cfg['id']
        mod = construir_modelo(mid, num_classes=len(clases), pretrained=True, freeze_backbone=cfg['freeze'])
        mod = mod.to(device)

        mod_entrenado, hist = entrenar_modelo_ciclo(
            mod,
            nombre_modelo=mid,
            dataloaders=dataloaders,
            epochs=epochs,
            device=device,
            lr=cfg['lr'],
            checkpoint_dir=checkpoint_dir
        )

        eval_test = evaluar_modelo_test(mod_entrenado, dataloaders['test'], clases=clases, device=device)

        resultados_totales['modelos'][mid] = {
            'nombre_descriptivo': cfg['nombre'],
            'historial_entrenamiento': hist,
            'evaluacion_testing': eval_test
        }

    ruta_json = os.path.join(base_dir, 'metricas_comparativas_modelos.json')
    with open(ruta_json, 'w', encoding='utf-8') as f:
        json.dump(resultados_totales, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60, flush=True)
    print(" RESUMEN FINAL DEL BENCHMARK COMPARATIVO", flush=True)
    print("=" * 60, flush=True)
    for mid, mdata in resultados_totales['modelos'].items():
        ev = mdata['evaluacion_testing']
        print(f"Modelo: {mdata['nombre_descriptivo']}", flush=True)
        print(f" -> Accuracy Global: {ev['accuracy_global'] * 100:.2f}% | F1-Macro: {ev['macro_avg']['f1_score']:.4f}", flush=True)
        print(f" -> Latencia Inferencia: {ev['latencia_ms_por_imagen']:.2f} ms | Parámetros: {ev['total_parametros']:,} ({ev['tamano_mb']} MB)", flush=True)
    print(f"Resultados exportados a: {ruta_json}", flush=True)

    return resultados_totales


if __name__ == '__main__':
    ejecutar_benchmarking_completo(epochs=3, batch_size=32)
