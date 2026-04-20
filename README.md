# Sistema Mixture of Experts para Clasificación Médica Multimodal

Sistema **Mixture of Experts (MoE) ciego a metadatos** que clasifica radiografías, imágenes dermoscópicas y tomografías 3D usando únicamente la información contenida en los píxeles. El sistema detecta automáticamente la modalidad del dato (2D vs. 3D), enruta la entrada al experto correspondiente y clasifica la patología. Incluye un *ablation study* formal de cuatro mecanismos de *routing* (ViT+Linear, GMM, Naive Bayes, k-NN) sobre los mismos embeddings.

> Proyecto final del curso **Incorporar Elementos de IA — Unidad II (Bloque Visión)** · Universidad Autónoma de Occidente.

---

## Tabla de contenido

- [Características](#características)
- [Arquitectura](#arquitectura)
- [Resultados](#resultados)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Datasets](#datasets)
- [Entrenamiento de los expertos](#entrenamiento-de-los-expertos)
- [Ablation study del router](#ablation-study-del-router)
- [Ejecución del sistema MoE completo](#ejecución-del-sistema-moe-completo)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Autores](#autores)
- [Referencias](#referencias)

---

## Características

- **Ciego a metadatos:** la única entrada es el tensor de la imagen o volumen. No se reciben etiquetas de modalidad, nombres de archivo ni ningún dato textual.
- **Preprocesador adaptativo:** detecta 2D (`rank=4`) vs. 3D (`rank=5`) y aplica el pipeline correspondiente.
- **5 expertos heterogéneos** especializados por dominio patológico.
- **Ablation study** de 4 mecanismos de *routing* sobre el mismo backbone ViT-Tiny congelado.
- **Balance de carga** con *Auxiliary Loss* del Switch Transformer y α dinámico.
- **Optimización de VRAM:** *gradient checkpointing*, FP16, *gradient accumulation* para entrenar en 12 GB.

---

## Arquitectura

```
┌────────────────────────────────────────────────────┐
│  Imagen o volumen (PNG / JPEG / NIfTI)             │
│  — sin metadatos —                                 │
└──────────────────────┬─────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  PREPROCESADOR ADAPTATIVO   │
        │  rank=4 → 2D resize 224²    │
        │  rank=5 → 3D resize 64³     │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   ROUTER  (ViT + Linear)    │
        │   embeddings ∈ ℝ¹⁹²         │
        └──────────────┬──────────────┘
                       │
   ┌────────┬──────────┼──────────┬────────┐
   │        │          │          │        │
┌──▼──┐ ┌──▼──┐   ┌───▼───┐   ┌──▼──┐  ┌──▼──┐
│ NIH │ │ISIC │   │Rodilla│   │LUNA │  │Pán- │
│Dense│ │Eff-B4│  │VGG-BN │   │R3D  │  │creas│
│ 121 │ │      │  │       │   │patch│  │R3D  │
└─────┘ └──────┘  └───────┘   └─────┘  └─────┘
  2D      2D         2D          3D       3D
```

### Expertos

| # | Dataset | Modelo | Entrada | Tarea |
|---|---------|--------|---------|-------|
| 1 | NIH ChestX-ray14 | DenseNet121 | 384×384 (2D) | 6 patologías torácicas (multietiqueta) |
| 2 | ISIC 2019 | EfficientNet-B4 | 260×260 (2D) | 8 clases dermoscópicas |
| 3 | Knee Osteoarthritis | VGG16-BN | 224×224 (2D) | Grados Kellgren-Lawrence (KL) |
| 4 | LUNA16 | R3D-18 (patch) | 32³ vóxeles | Nódulo / no-nódulo |
| 5 | PANORAMA (Páncreas) | R3D-18 (1-canal) | 64³ vóxeles | PDAC / no-PDAC |

---

## Resultados

### F1 Macro por experto

| Experto | F1 Macro | Objetivo | Estado |
|---------|---------:|---------:|:------:|
| NIH (2D) | 0.5322 | 0.72 | ✗ |
| ISIC (2D) | 0.7421 | 0.72 | ✓ |
| Rodilla (2D) | 0.9500 | 0.72 | ✓ |
| LUNA16 (3D) | 0.7479 | 0.65 | ✓ |
| Páncreas (3D) | 0.7612 | 0.65 | ✓ |

### Ablation study del router (ViT-Tiny congelado, d=192)

| Router | Tipo | Routing Acc. | Latencia | Gradiente |
|--------|------|-------------:|---------:|:---------:|
| ViT + Linear | Paramétrico (gradiente) | 0.9924 | ~0 ms | ✓ |
| ViT + GMM | Paramétrico (EM) | 0.5978 | 0.03 ms | ✗ |
| ViT + Naive Bayes | Paramétrico (MLE) | 0.9997 | 0.01 ms | ✗ |
| ViT + k-NN | No paramétrico | 1.0000 | 0.29 ms | ✗ |

### Balance de carga

Con α dinámico (base = 0.10) y *Auxiliary Loss* del Switch Transformer, el ratio `max(fᵢ)/min(fᵢ)` convergió a **1.005** desde la época 3, muy por debajo del umbral crítico de 1.30.

---

## Estructura del repositorio


```
.
├── notebooks/
│   ├── entrenamiento_nih.ipynb          # Experto 1 (DenseNet121)
│   ├── entrenamiento_isic.ipynb         # Experto 2 (EfficientNet-B4)
│   ├── entrenamiento_rodilla.ipynb      # Experto 3 (VGG16-BN)
│   ├── entrenamiento_luna.ipynb         # Experto 4 (R3D-18 patch)
│   └── entrenamiento_pancreatic.ipynb   # Experto 5 (R3D-18)
├── moe/
│   ├── moe_definitivo.ipynb             # Implementacion y entrenamiento de MoE
├── ablation/
│   └── router_ablation.ipynb            # Comparación de los 4 mecanismos
└── README.md
```

---

## Requisitos

- Python 3.10+
- CUDA 11.8 o 12.x con GPU de ≥12 GB de VRAM (probado en RTX 3060 12GB y RTX 4090 24GB)
- ~30 GB de almacenamiento libre para datasets + cache

### Dependencias principales

```
torch>=2.1
torchvision>=0.16
timm>=0.9
scikit-learn>=1.3
opencv-python
pandas
Pillow
torchxrayvision
nibabel           # para NIfTI (LUNA, Páncreas)
SimpleITK         # resampling 3D
faiss-cpu         # k-NN router
matplotlib
seaborn
tqdm
```

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate       # Linux/Mac
# .\venv\Scripts\activate      # Windows

```

---

## Datasets

Los datasets **no se incluyen** en el repositorio por tamaño y licencia. Deben descargarse por separado:

| Dataset | Tamaño aprox. | Enlace |
|---------|--------------:|--------|
| NIH ChestX-ray14 | ~45 GB | [nihcc.app.box.com](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| ISIC 2019 | ~9 GB | [challenge.isic-archive.com](https://challenge.isic-archive.com/data/#2019) |
| Knee Osteoarthritis | ~1 GB | [Kaggle — Knee OA](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) |
| LUNA16 | ~120 GB | [luna16.grand-challenge.org](https://luna16.grand-challenge.org/) |
| PANORAMA (Páncreas) | ~25 GB | [panorama.grand-challenge.org](https://panorama.grand-challenge.org/) |

Estructura esperada:

```
DATA/
├── nih/
│   ├── Data_Entry_2017.csv
│   └── data384/          # imágenes ya redimensionadas a 384px
├── isic/
├── knee/
├── luna16/
│   ├── annotations.csv
│   ├── candidates.csv
│   └── pixels/           # _pixels.npy + _spacing.npy
└── panorama/
    └── cache_pt/         # tensores .pt preprocesados (ver notebook)
```

---

## Entrenamiento de los expertos

Cada experto se entrena de forma independiente en su notebook. El orden recomendado es:

1. **Rodilla** (el más rápido, ~30 min) — sirve para verificar que el pipeline 2D funciona.
2. **ISIC** (~2 h) — valida el pipeline con desbalance moderado.
3. **NIH** (~6-8 h) — el más sensible; ver [Limitaciones](#limitaciones-conocidas).
4. **LUNA16** (~3-4 h) — requiere pre-extracción de parches 32³.
5. **Páncreas** (~1 h de entrenamiento + 1-3 h de conversión offline `.nii.gz → .pt`).

```bash
# Ejemplo: ejecutar el notebook de LUNA16
jupyter lab notebooks/entrenamiento_luna.ipynb
```

Al finalizar cada notebook se guardan los pesos en `checkpoints/experto_<nombre>_best.pt`.

---

## Ablation study del router

El *ablation* compara los cuatro mecanismos de *routing* sobre los **mismos** embeddings del CLS token de ViT-Tiny. Lo único que cambia entre experimentos es la cabeza de decisión.

```bash
jupyter lab ablation/router_ablation.ipynb
```

El notebook:

1. Extrae embeddings de los 5 dominios con ViT-Tiny congelado.
2. Entrena y evalúa los cuatro routers con los mismos *splits*.
3. Reporta *Routing Accuracy*, latencia y matriz de confusión por router.

---

## Ejecución del sistema MoE completo

```python
from moe.moe_system import MoESystem

# Cargar el sistema con todos los expertos entrenados
moe = MoESystem.from_pretrained("checkpoints/")

# Clasificar una imagen 2D (radiografía, dermatoscopia, rodilla)
result = moe.predict("ruta/a/imagen.png")

# Clasificar un volumen 3D (CT pulmonar o pancreático)
result = moe.predict("ruta/a/volumen.nii.gz")

# result incluye:
#   - expert_used: qué experto decidió el router
#   - routing_probs: probabilidades del router sobre los 5 expertos
#   - prediction: clase predicha por el experto
#   - confidence: probabilidad de la predicción
```

---

## Limitaciones conocidas

- **Experto NIH:** no alcanzó el umbral (0.53 vs. 0.72 objetivo). Se probaron Swin-Tiny, ConvNeXt-Tiny, Asymmetric Loss, Mixup multietiqueta y resoluciones bajas; ninguna superó la configuración final (DenseNet121 a 384 px con CLAHE). Las etiquetas ruidosas derivadas de NLP sobre informes radiológicos son la limitación estructural principal.
- **Router GMM:** falló por inestabilidad al ajustar matrices de covarianza completas (192×192) con las muestras disponibles por dominio. Una configuración `covariance_type='diag'` habría sido más robusta.
- **Páncreas:** el dataset (~445 volúmenes) es demasiado pequeño para explotar R3D-18; un preentrenamiento en MedicalNet sería recomendable.
- **Hardware:** los expertos 3D requieren *gradient checkpointing* y FP16 para caber en 12 GB de VRAM.

---

## Autores

- **Juan Jose Orozco Lopez**
- **David Melo Valbuena**
- **Gabriel Eduardo Martinez Martinez**
- **Juan Jacobo Delgado Melo**

Analítica de Datos · Universidad Autónoma de Occidente. Santiago de Cali | 
Ing. de Datos e IA 

---

## Referencias

Las referencias completas están en el reporte técnico (`report/proyecto_moe_final.tex`). Referencias principales:

- Fedus, Zoph & Shazeer. *Switch Transformers*. JMLR, 2022.
- Jacobs, Jordan, Nowlan & Hinton. *Adaptive Mixtures of Local Experts*. Neural Computation, 1991.
- Setio et al. *LUNA16 Challenge*. Medical Image Analysis, 2017.
- Zhu et al. *DeepLung*. WACV, 2018.
- Chen, Ma & Zheng. *Med3D*. arXiv:1904.00625, 2019.
- Huang et al. *DenseNet*. CVPR, 2017.
- Tan & Le. *EfficientNet*. ICML, 2019.

---

## Licencia

Uso académico. Los datasets mantienen sus licencias originales; verificar cada fuente antes de redistribuir resultados o modelos derivados.
