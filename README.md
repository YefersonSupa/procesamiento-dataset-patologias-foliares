<div align="center">

# Detección y Clasificación de Patologías Foliares en Papa (*Solanum tuberosum*) mediante Visión Artificial y Deep Learning

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Universidad Nacional de San Antonio Abad del Cusco (UNSAAC)**  
*Facultad de Ingeniería Eléctrica, Electrónica, Informática y Mecánica*  
*Escuela Profesional de Ingeniería Informática y de Sistemas*  
**Asignatura:** Visión Computacional (IF657BIN) — Semestre 2026-II  
**Docente:** Ing. CCACYAHUILLCA BEJAR, Hans Harley  

</div>

---

## 📋 Tabla de Contenidos
- [1. Descripción y Justificación del Proyecto](#1-descripción-y-justificación-del-proyecto)
- [2. Dataset Extendido (4,072 Imágenes)](#2-dataset-extendido-4072-imágenes)
- [3. Pipeline de Preprocesamiento de Imágenes](#3-pipeline-de-preprocesamiento-de-imágenes)
- [4. Comparación de 3 Arquitecturas Convolucionales](#4-comparación-de-3-arquitecturas-convolucionales)
- [5. Resultados Cuantitativos del Benchmarking](#5-resultados-cuantitativos-del-benchmarking)
- [6. Matrices de Confusión y Análisis de Error](#6-matrices-de-confusión-y-análisis-de-error)
- [7. Estructura del Repositorio](#7-estructura-del-repositorio)
- [8. Instalación y Requisitos](#8-instalación-y-requisitos)
- [9. Guía de Ejecución y Reproducibilidad](#9-guía-de-ejecución-y-reproducibilidad)
- [10. Pruebas Unitarias](#10-pruebas-unitarias)
- [11. Entregables Académicos y Documentación](#11-entregables-académicos-y-documentación)
- [12. Autores y Créditos](#12-autores-y-créditos)
- [13. Licencia](#13-licencia)

---

## 1. Descripción y Justificación del Proyecto

El cultivo de la papa (*Solanum tuberosum*) representa la columna vertebral de la economía agrícola andina en el departamento del Cusco y el Perú. Sin embargo, las condiciones microclimáticas de humedad y temperatura favorecen el desarrollo de devastadoras enfermedades fitopatológicas:

1. **Tizón Temprano (*Early Blight*):** Producido por el hongo fitopatógeno *Alternaria solani*, caracterizado por manchas concéntricas en diana y halos cloróticos.
2. **Tizón Tardío (*Late Blight*):** Causado por el oomiceto *Phytophthora infestans*, una enfermedad de diseminación rápida que causa lesiones foliares acuosas oscuras que necrosan el follaje en pocos días.
3. **Follaje Sano (*Healthy*):** Hojas de control fisiológicamente sanas, sin signos de estrés biótico.

Este proyecto semestral implementa una solución integral de **Visión por Computador y Aprendizaje Profundo (Deep Learning)** para automatizar el diagnóstico temprano, comparando rigurosamente tres paradigmas convolucionales (diseño propio desde cero, aprendizaje residual profundo y convoluciones separables ligeras para dispositivos móviles).

---

## 2. Dataset Extendido (4,072 Imágenes)

Para la **Segunda Entrega**, el banco de datos se expandió incorporando el dataset completo de **4,072 imágenes foliares** a resolución estándar $256 \times 256$ píxeles, estructurado en particiones estratificadas:

| Partición | Early Blight | Healthy | Late Blight | Total Imágenes | Porcentaje |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Training** | 1,303 | 816 | 1,132 | **3,251** | 79.8% |
| **Validation** | 163 | 102 | 151 | **416** | 10.2% |
| **Testing (Independiente)** | 162 | 102 | 141 | **405** | 10.0% |
| **Total Global** | **1,628** | **1,020** | **1,424** | **4,072** | **100.0%** |

* **Auditoría de Nitidez:** Se aplicó la varianza del operador Laplaciano discreto $\text{Var}(\nabla^2 I) \ge 80.0$, certificando que el **94.2%** de las muestras presentan calidad óptima libre de desenfoque por movimiento.

---

## 3. Pipeline de Preprocesamiento de Imágenes

Cada imagen foliar es acondicionada mediante un pipeline determinista en `src/preprocesamiento.py`:

```
Imagen Cruda (BGR) ──► Filtrado Bilateral ──► CLAHE en CIE-Lab ──► Letterbox (256x256) ──► Data Augmentation (Train)
```

1. **Filtrado Espacial Bilateral:** Atenúa el ruido de alta frecuencia del sensor fotográfico conservando con precisión subpíxel los bordes y gradientes de las lesiones necróticas ($d=5, \sigma_s=50, \sigma_r=50$).
2. **Realce Adaptativo de Contraste Local (CLAHE en CIE-Lab):** Desacopla la luminancia ($L^*$) de los canales cromáticos ($a^*, b^*$). Aplica ecualización adaptativa exclusivamente sobre $L^*$ con `clipLimit=2.0` y teselas de $8 \times 8$, evitando sobresaturar los matices naturales de la clorofila.
3. **Estandarización Espacial con Letterbox:** Redimensiona isotrópicamente a $256 \times 256$ rellenando bordes simétricos constantes, preservando la relación de aspecto original (1:1) sin deformaciones anatómicas.
4. **Aumento de Datos en Entrenamiento:** Rotaciones controladas ($\pm 15^\circ$), volteos horizontales y ajustes de brillo/contraste.

---

## 4. Comparación de 3 Arquitecturas Convolucionales

Se diseñaron e implementaron tres arquitecturas en PyTorch para contrastar sus fortalezas:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       COMPARATIVA DE ARQUITECTURAS                          │
├───────────────────────┬─────────────────────────────┬───────────────────────┤
│ 1. Custom Potato CNN  │ 2. ResNet-18 (Residual)     │ 3. MobileNet-V2 (Edge)│
├───────────────────────┼─────────────────────────────┼───────────────────────┤
│ • Diseñada desde cero │ • Conexiones Skip residuales│ • Depthwise Separable │
│ • 3 bloques Conv-BN   │ • Pre-entrenada en ImageNet │ • Inverted Bottlenecks│
│ • 645,827 Parámetros  │ • 11,178,051 Parámetros     │ • 2,227,651 Parámetros│
│ • Tamaño: 2.46 MB     │ • Tamaño: 44.71 MB          │ • Tamaño: 8.91 MB     │
│ • Baseline referencial│ • Máxima capacidad diagnóst.│ • Óptima para móviles │
└───────────────────────┴─────────────────────────────┴───────────────────────┘
```

---

## 5. Resultados Cuantitativos del Benchmarking

Evaluación independiente sobre las **405 imágenes** del split de **Testing**:

| Modelo | Exactitud (Acc) | F1-Score Macro | Parámetros | Tamaño Disco | Latencia Inferencia (CPU) | Recomendación de Uso |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Custom Potato CNN** | 66.17% | 0.6462 | 618,787 | 2.36 MB | 10.50 ms/img | Baseline de Control |
| **ResNet-18 (Transfer)** | 77.04% | 0.7596 | 11,178,051 | 42.64 MB | 21.24 ms/img | Servidor Central / Laboratorio |
| **MobileNet-V2 (Edge)** | **80.74%** | **0.7828** | **2,227,715** | **8.50 MB** | **10.36 ms/img** | **App Móvil en Campo / Dron** |

### Conclusiones del Benchmarking:
* **Transfer Learning:** Aporta un incremento de $+14.57\%$ en exactitud sobre el modelo entrenado desde cero (MobileNet-V2 con 80.74% vs Custom CNN con 66.17%).
* **Compromiso Óptimo para Campo:** **MobileNet-V2** reduce los parámetros en un **80%** respecto a ResNet-18 y disminuye la latencia a **10.36 ms en CPU**, logrando la exactitud más alta del benchmark.

---

## 6. Matrices de Confusión y Análisis de Error

* **Follaje Sano (*Healthy*):** 100% de especificidad en ResNet-18 ($F_1 = 1.000$) y 99.0% en MobileNet-V2.
* **Confusión Inter-Clase:** Las únicas discrepancias se presentaron entre *Early Blight* y *Late Blight* en fases tempranas, donde las microlesiones foliares carecen aún de los anillos concéntricos maduros.

---

## 7. Estructura del Repositorio

```
procesamiento_dataset/
├── .github/workflows/ci.yml           # Integración continua automatizada
├── assets/
│   ├── pipeline_comparativa.png       # Visualización de fases del pipeline
│   ├── eda_report.png                 # Dashboard de análisis exploratorio
│   └── figuras_modelos/               # Curvas, matrices y trade-off de modelos
├── data_raw/
│   ├── Early_Blight/                  # Muestras crudas de control
│   ├── Healthy/                       # Muestras crudas de control
│   ├── Late_Blight/                   # Muestras crudas de control
│   └── README.md                      # Documentación del origen de los datos
├── docs/
│   ├── unsaac.jpg                     # Logo oficial UNSAAC
│   ├── articulo_cientifico_u2.tex      # Artículo científico formal en LaTeX (IEEE 2 cols)
│   ├── referencias_u2.bib             # Referencias bibliográficas BibTeX
│   ├── presentacion_10min_u2.tex      # Diapositivas Beamer (16:9) para 10 min
│   ├── portal_interactivo_u2.html     # Portal interactivo web con KaTeX
│   └── portal_interactivo_u2.pdf      # PDF compilado para entrega inmediata
├── notebooks/
│   ├── Primera_Entrega_Preprocesamiento_Dataset.ipynb
│   └── Segunda_Entrega_Benchmarking_Modelos.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocesamiento.py            # Funciones de filtrado, CLAHE y letterbox
│   ├── pipeline.py                    # Pipeline de particionado y procesamiento
│   ├── extender_dataset.py            # Ingesta, descompresión y censo de datos
│   ├── modelos_u2.py                  # Arquitecturas PyTorch (Custom, ResNet, MobileNet)
│   ├── entrenamiento_benchmark.py     # Ciclos de entrenamiento y evaluación en test
│   └── generar_figuras_u2.py         # Gráficas científicas y matrices de confusión
├── tests/
│   ├── __init__.py
│   ├── test_preprocesamiento.py       # Pruebas unitarias de visión por computador
│   └── test_modelos_u2.py             # Pruebas unitarias de modelos y dimensiones
├── metadata_dataset_u2.json           # Censo estadístico del dataset extendido
├── metricas_comparativas_modelos.json  # Métricas cuantitativas de benchmarking
├── requirements.txt                   # Dependencias del proyecto
├── pyproject.toml                     # Configuración del paquete
├── LICENSE                            # Licencia MIT
└── README.md                          # Este documento
```

---

## 8. Instalación y Requisitos

Se recomienda utilizar un entorno virtual (venv o conda):

```bash
# 1. Clonar el repositorio
git clone https://github.com/YefersonSupa/procesamiento-dataset-patologias-foliares.git
cd procesamiento-dataset-patologias-foliares

# 2. Crear y activar entorno virtual
python -m venv venv
# En Linux/macOS: source venv/bin/activate
# En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 9. Guía de Ejecución y Reproducibilidad

### Paso 1: Extracción y Auditoría del Dataset Extendido
```bash
python src/extender_dataset.py
```

### Paso 2: Ejecución del Benchmarking de los 3 Modelos
```bash
python src/entrenamiento_benchmark.py
```

### Paso 3: Generación de Figuras Científicas
```bash
python src/generar_figuras_u2.py
```

---

## 10. Pruebas Unitarias

El proyecto cuenta con una suite completa de pruebas unitarias con cobertura sobre el pipeline de visión y las dimensiones de entrada/salida de las redes:

```bash
python -m unittest discover -s tests -v
```

Resultado esperado:
```
Ran 14 tests in 1.528s
OK
```

---

## 11. Entregables Académicos y Documentación

1. **Artículo Científico en LaTeX:** [`docs/articulo_cientifico_u2.tex`](docs/articulo_cientifico_u2.tex) (Formato 2 columnas estilo IEEEtran con carátula institucional reglamentaria UNSAAC).
2. **Presentación de 10 Minutos:** [`docs/presentacion_10min_u2.tex`](docs/presentacion_10min_u2.tex) (Diapositivas Beamer 16:9 estructuradas cronométricamente).
3. **Portal Web Interactivo (KaTeX):** [`docs/portal_interactivo_u2.html`](docs/portal_interactivo_u2.html) (Plataforma interactiva con botón de exportación PDF A4).
4. **Informe en PDF Compilado:** [`docs/portal_interactivo_u2.pdf`](docs/portal_interactivo_u2.pdf).
5. **Notebook Jupyter:** [`notebooks/Segunda_Entrega_Benchmarking_Modelos.ipynb`](notebooks/Segunda_Entrega_Benchmarking_Modelos.ipynb).

---

## 12. Autores y Créditos

* **SUPA CUSIPAUCAR, Yeferson** — Código: `220553` — [GitHub: @YefersonSupa](https://github.com/YefersonSupa)
* **VILLAVICENCIO SEGUIL, Edu Piero** — Código: `220555`

**Docente:** Ing. CCACYAHUILLCA BEJAR, Hans Harley  
**Institución:** Universidad Nacional de San Antonio Abad del Cusco (UNSAAC)

---

## 13. Licencia

Este proyecto está bajo la Licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
