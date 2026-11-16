"""
Generador de Figuras Científicas y Dashboards de Visualización (Unidad 2)
Proyecto Semestral: Comparación de 3 Modelos en Visión Artificial
Asignatura: Visión Computacional (IF657BIN)
Docente: CCACYAHUILLCA BEJAR, HANS HARLEY
Autores:
  - SUPA CUSIPAUCAR, Yeferson - 220553
  - VILLAVICENCIO SEGUIL, Edu Piero - 220555
UNSAAC - 2026-2
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


def configurar_estilo():
    """Configura el estilo estético para publicaciones científicas."""
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#EEEEEE'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7


def generar_curvas_entrenamiento(datos_modelos: Dict[str, Any], ruta_salida: str):
    """Genera gráfico comparativo de curvas de pérdida y precisión por época."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    colores = {'custom_cnn': '#E63946', 'resnet18': '#1D3557', 'mobilenetv2': '#2A9D8F'}
    estilos = {'custom_cnn': 'o--', 'resnet18': 's-', 'mobilenetv2': '^:'}

    # Subplot 1: Train & Val Loss
    ax_loss = axes[0]
    for mid, mdata in datos_modelos.items():
        hist = mdata['historial_entrenamiento']
        epochs = range(1, len(hist['train_loss']) + 1)
        ax_loss.plot(epochs, hist['train_loss'], estilos[mid], label=f"{mdata['nombre_descriptivo']} (Train)", color=colores[mid], alpha=0.6)
        ax_loss.plot(epochs, hist['val_loss'], estilos[mid], label=f"{mdata['nombre_descriptivo']} (Val)", color=colores[mid], linewidth=2.0)
    
    ax_loss.set_title("Evolución de la Función de Pérdida (Cross-Entropy Loss)", fontsize=12, fontweight='bold', pad=12)
    ax_loss.set_xlabel("Épocas de Entrenamiento", fontsize=10)
    ax_loss.set_ylabel("Pérdida (Loss)", fontsize=10)
    ax_loss.grid(True)
    ax_loss.legend(fontsize=8, loc='upper right')

    # Subplot 2: Train & Val Accuracy
    ax_acc = axes[1]
    for mid, mdata in datos_modelos.items():
        hist = mdata['historial_entrenamiento']
        epochs = range(1, len(hist['train_acc']) + 1)
        ax_acc.plot(epochs, [a * 100 for a in hist['train_acc']], estilos[mid], label=f"{mdata['nombre_descriptivo']} (Train)", color=colores[mid], alpha=0.6)
        ax_acc.plot(epochs, [a * 100 for a in hist['val_acc']], estilos[mid], label=f"{mdata['nombre_descriptivo']} (Val)", color=colores[mid], linewidth=2.0)

    ax_acc.set_title("Evolución de la Exactitud (Accuracy %)", fontsize=12, fontweight='bold', pad=12)
    ax_acc.set_xlabel("Épocas de Entrenamiento", fontsize=10)
    ax_acc.set_ylabel("Exactitud (%)", fontsize=10)
    ax_acc.grid(True)
    ax_acc.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(ruta_salida, bbox_inches='tight')
    plt.close()
    print(f"[FIGURA] Curvas de entrenamiento guardadas en: {ruta_salida}")


def generar_matrices_confusion_comparativas(datos_modelos: Dict[str, Any], clases: list, ruta_salida: str):
    """Genera matrices de confusión lado a lado con heatmaps normalizados."""
    n_mods = len(datos_modelos)
    fig, axes = plt.subplots(1, n_mods, figsize=(5 * n_mods, 4.5), dpi=300)
    if n_mods == 1:
        axes = [axes]

    clases_limpias = [c.replace('_', ' ') for c in clases]

    for idx, (mid, mdata) in enumerate(datos_modelos.items()):
        cm = np.array(mdata['evaluacion_testing']['matriz_confusion'])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        ax = axes[idx]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=clases_limpias, yticklabels=clases_limpias if idx == 0 else False)
        
        acc = mdata['evaluacion_testing']['accuracy_global'] * 100
        ax.set_title(f"{mdata['nombre_descriptivo']}\n(Acc Global: {acc:.1f}%)", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Predicción del Modelo", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Clase Real (Ground Truth)", fontsize=9)

    plt.tight_layout()
    plt.savefig(ruta_salida, bbox_inches='tight')
    plt.close()
    print(f"[FIGURA] Matrices de confusión guardadas en: {ruta_salida}")


def generar_comparativa_metricas_barras(datos_modelos: Dict[str, Any], clases: list, ruta_salida: str):
    """Genera gráfico de barras con F1-Score por clase y métricas agregadas."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    ancho = 0.25
    x = np.arange(len(clases) + 2)  # 3 clases + Macro-F1 + Accuracy Global
    etiquetas = [c.replace('_', ' ') for c in clases] + ['Macro F1', 'Accuracy Global']
    
    colores = {'custom_cnn': '#E63946', 'resnet18': '#1D3557', 'mobilenetv2': '#2A9D8F'}
    
    for idx, (mid, mdata) in enumerate(datos_modelos.items()):
        ev = mdata['evaluacion_testing']
        f1_vals = [ev['metricas_por_clase'][c]['f1_score'] * 100 for c in clases]
        macro_f1 = ev['macro_avg']['f1_score'] * 100
        acc_glob = ev['accuracy_global'] * 100
        valores = f1_vals + [macro_f1, acc_glob]
        
        offset = (idx - 1) * ancho
        barras = ax.bar(x + offset, valores, width=ancho, label=mdata['nombre_descriptivo'], color=colores[mid], alpha=0.9)
        
        for bar in barras:
            altura = bar.get_height()
            ax.annotate(f"{altura:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, altura),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_title("Comparativa de Rendimiento por Clase y Métricas Globales (%)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=9, fontweight='medium')
    ax.set_ylabel("Puntuación (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(ruta_salida, bbox_inches='tight')
    plt.close()
    print(f"[FIGURA] Comparativa de barras guardada en: {ruta_salida}")


def generar_tradeoff_eficiencia(datos_modelos: Dict[str, Any], ruta_salida: str):
    """Genera gráfico de dispersión / trade-off entre Exactitud vs Latencia vs Tamaño."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    colores = {'custom_cnn': '#E63946', 'resnet18': '#1D3557', 'mobilenetv2': '#2A9D8F'}

    for mid, mdata in datos_modelos.items():
        ev = mdata['evaluacion_testing']
        latencia = ev['latencia_ms_por_imagen']
        acc = ev['accuracy_global'] * 100
        tam_mb = ev['tamano_mb']
        params_m = ev['total_parametros'] / 1e6

        # Tamaño del marcador proporcional al tamaño en disco
        scatter_size = max(100, min(800, tam_mb * 15))
        ax.scatter(latencia, acc, s=scatter_size, color=colores[mid], alpha=0.7, edgecolors='black', linewidth=1.5)
        
        texto_label = f"{mdata['nombre_descriptivo']}\n({params_m:.2f}M params, {tam_mb} MB, {latencia:.1f} ms)"
        ax.annotate(texto_label, xy=(latencia, acc), xytext=(10, -5), textcoords='offset points',
                    fontsize=8, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colores[mid], alpha=0.8))

    ax.set_title("Trade-off: Exactitud vs Latencia de Inferencia en CPU", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Latencia de Inferencia (ms por imagen)", fontsize=10)
    ax.set_ylabel("Exactitud Global en Testing (%)", fontsize=10)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(ruta_salida, bbox_inches='tight')
    plt.close()
    print(f"[FIGURA] Trade-off de eficiencia guardado en: {ruta_salida}")


def ejecutar_generacion_todas_figuras():
    """Carga metricas_comparativas_modelos.json y exporta todas las figuras."""
    configurar_estilo()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_json = os.path.join(base_dir, 'metricas_comparativas_modelos.json')
    dir_salida = os.path.join(base_dir, 'assets', 'figuras_modelos')
    os.makedirs(dir_salida, exist_ok=True)

    if not os.path.exists(ruta_json):
        print(f"[ERROR] Archivo {ruta_json} no encontrado. Ejecuta primero entrenamiento_benchmark.py")
        return

    with open(ruta_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)

    modelos = datos['modelos']
    clases = datos['clases']

    generar_curvas_entrenamiento(modelos, os.path.join(dir_salida, 'curvas_entrenamiento_comparativa.png'))
    generar_matrices_confusion_comparativas(modelos, clases, os.path.join(dir_salida, 'matrices_confusion_comparativa.png'))
    generar_comparativa_metricas_barras(modelos, clases, os.path.join(dir_salida, 'comparativa_barras_metricas.png'))
    generar_tradeoff_eficiencia(modelos, os.path.join(dir_salida, 'tradeoff_eficiencia_rendimiento.png'))
    print("[OK] Todas las figuras científicas fueron generadas con éxito.")


if __name__ == '__main__':
    ejecutar_generacion_todas_figuras()
