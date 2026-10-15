# Dataset Foliar Crudo (*Raw Data*)

Este directorio contiene las muestras de imágenes foliares crudas organizadas por clases patológicas para el cultivo de papa (*Solanum tuberosum*).

## Clases Representadas

| Clase | Patógeno Causante | Descripción Visual |
| :--- | :--- | :--- |
| **`Early_Blight`** | *Alternaria solani* | Manchas necróticas concéntricas oscuras con halo clorótico ("anillos concéntricos" o patrón en diana). |
| **`Late_Blight`** | *Phytophthora infestans* | Lesiones foliares acuosas pardas/negras de rápida expansión que necrosan el tejido foliar. |
| **`Healthy`** | Ninguno (Hoja Sana) | Hojas limpias y libres de necrosis foliar con textura y tonalidad verde natural. |

## Muestras Incluidas y Dataset Completo

- El repositorio incluye un conjunto de muestras ligeras para permitir la ejecución inmediata del pipeline y pruebas locales sin descargas adicionales.
- Para entrenar modelos a gran escala con el dataset completo (ej. PlantVillage Potato Dataset), descargue las imágenes completas desde Kaggle o Zenodo y distribúyalas dentro de sus respectivas subcarpetas:
  ```text
  data_raw/
  ├── Early_Blight/
  ├── Healthy/
  └── Late_Blight/
  ```
