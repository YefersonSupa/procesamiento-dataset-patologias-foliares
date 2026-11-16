# Dataset Foliar Crudo (*Raw Data*)

Este directorio contiene las muestras de imágenes foliares crudas organizadas por clases patológicas para el cultivo de papa (*Solanum tuberosum*).

## Fuente Oficial del Dataset (Kaggle)

* **Repositorio en Kaggle:** [Potato Disease Leaf Dataset (PLD)](https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld?resource=download)
* **Autor / Creador:** Md. Rizwan (`@rizwan123456789`)
* **Resolución Nativa:** $256 \times 256$ píxeles (formato RGB / JPG).
* **Volumen Total:** 4,072 imágenes foliares distribuidas en 3 clases patológicas.
* **Estructura Oficial:** Partición estándar en subdirectorios `Training` (3,251 imgs), `Validation` (416 imgs) y `Testing` (405 imgs).

## Clases Representadas

| Clase | Patógeno Causante | Descripción Visual |
| :--- | :--- | :--- |
| **`Early_Blight`** | *Alternaria solani* | Manchas necróticas concéntricas oscuras con halo clorótico ("anillos concéntricos" o patrón en diana). |
| **`Late_Blight`** | *Phytophthora infestans* | Lesiones foliares acuosas pardas/negras de rápida expansión que necrosan el tejido foliar. |
| **`Healthy`** | Ninguno (Hoja Sana) | Hojas limpias y libres de necrosis foliar con textura y tonalidad verde natural. |

## Muestras Incluidas y Dataset Completo

- El repositorio incluye un conjunto de muestras ligeras para permitir la ejecución inmediata del pipeline y pruebas locales sin descargas adicionales.
- Para reproducir el entrenamiento a gran escala con las 4,072 imágenes, descargue el archivo comprimido desde el enlace de Kaggle o utilice el módulo:
  ```bash
  python src/extender_dataset.py
  ```
  lo cual descomprime e integra `archive.zip` (`PLD_3_Classes_256`) en la estructura del proyecto.
