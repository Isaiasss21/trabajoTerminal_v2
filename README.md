# TrabajoTerminal_v2

## Visión general

Este repositorio implementa un flujo completo de procesamiento de microexpresiones faciales, desde la extracción de secuencias de frames hasta el entrenamiento de un modelo de clasificación de emociones basado en secuencias de características.

El proyecto se organiza en tres capas principales:

- `scripts/`: scripts de extracción y unión de datasets.
- `src/`: código de preprocesamiento, definición de modelos y entrenamiento.
- `outputs_dataset_index/`: índices CSV de secuencias de video ya preparados.

## Estructura del repositorio

- `scripts/01_extract_resnet.py`: extrae embeddings de ResNet de secuencias de frames, usando detección de cara y búsqueda de apex.
- `scripts/02_unir_dataset.py`: combina índices de caracteristicas extraídas de dos datasets base en un dataset maestro.
- `src/preprocessing.py`: clase `PreprocesadorFacial` que detecta caras, landmarks y calcula features sobre secuencias.
- `src/model.py`: define modelos de entrenamiento, incluyendo LSTM con atención temporal y un encoder de frames con ResNet-18.
- `src/dataio.py`: funciones utilitarias para leer/guardar índices CSV y convertir rutas de secuencias.
- `src/train.py`: implementa dataset, normalización, entrenamiento, validación y evaluación.
- `outputs_dataset_index/`: contiene archivos CSV con rutas de secuencias del dataset.

## Datos de entrada

El repositorio maneja un índice de secuencias en formato CSV con estas columnas esenciales:

- `persona`: identificador del sujeto.
- `camara`: etiqueta de la cámara o fuente.
- `emocion`: etiqueta de emoción.
- `paths_joined`: cadenas de rutas de imágenes separadas por `|`.

Ejemplo de `outputs_dataset_index/sequences.csv`:

- `p003,camaraWeb,Asco,D:\\datasetVideosTT\\03_grabaciones\\p003\\camaraWeb\\Asco\\framesWeb\\frame_3500.jpg|...`

## Pipeline de procesamiento

### 1. Extracción de características spatial-temporales (`scripts/01_extract_resnet.py`)

Este script procesa cada secuencia de frames para extraer un conjunto de características en formato `.npy`:

- Carga un CSV con secuencias y rutas de frames.
- Detecta la cara en cada frame usando YOLOv8-face.
- Extrae landmarks faciales con `dlib.shape_predictor_68_face_landmarks.dat`.
- Localiza el frame de mayor cambio respecto a una baseline inicial para seleccionar el apex.
- Realiza alineación o crop de la cara y redimensiona al tamaño que espera ResNet-18.
- Calcula embeddings de ResNet-18 pre-entrenado en ImageNet.
- Guarda cada secuencia como un archivo `.npy` de embeddings y registra su ruta en `features_index_resnet.csv`.

Salida esperada:

- Carpeta de salida `pipeline_out_resnet/features_resnet/` con archivos `.npy`.
- Archivo `pipeline_out_resnet/features_index_resnet.csv` con columnas:
  - `persona`
  - `camara`
  - `emocion`
  - `feature_path`
  - `num_frames`

### 2. Unión de datasets (`scripts/02_unir_dataset.py`)

Este script combina varios archivos CSV de características en un único dataset maestro.

Funciona así:

- Lee `pipeline_out_resnet_casme/features_index_resnet.csv` si existe.
- Lee `pipeline_out_resnet_smic/features_index_resnet.csv` si existe.
- Concatena ambos DataFrames en uno solo.
- Guarda el resultado en `pipeline_out_resnet_maestro/features_index_maestro.csv`.

Este paso permite entrenar con un dataset combinado que integra muestras de diferentes orígenes.

## Componentes de `src/`

### `src/preprocessing.py`

Define la clase `PreprocesadorFacial` y utilidades de descarga.

Funciones importantes:

- `ensure_yolo_face_model(dst)`: descarga el modelo `yolov8n-face.pt` si no existe.
- `ensure_dlib_shape_predictor(dst)`: descarga y descomprime `shape_predictor_68_face_landmarks.dat`.
- `PreprocesadorFacial`:
  - inicializa YOLOv8 para detección de caras.
  - inicializa `dlib` para landmarks faciales.
  - normaliza landmarks dentro de la caja de la cara.
  - calcula características de optical flow entre frames en la región de interés.

Uso clave:

- Detección de cara y landmarks.
- Extracción de ROI facial alineada.
- Cálculo de cambios temporales en la región de la cara.

### `src/model.py`

Contiene definiciones de modelos y bloques de red.

Modelos principales:

- `TemporalAttention`: atención temporal sobre secuencias output de la LSTM.
- `EmotionLSTM`: red recurrente para clasificación de emociones.
  - LSTM bidireccional opcional.
  - atención temporal sobre las salidas de la LSTM.
  - normalización de baseline opcional usando primeros frames.
  - clasificador final con capas lineales y dropout.
- `ResidualMaskingGate`: módulo de máscara residual útil para refinamiento de características.
- `FrameEncoderResNet18`: encoder basado en ResNet-18.
  - conserva la parte convolucional de ResNet-18.
  - aplica una máscara residual ligera antes de proyección.
  - opcionalmente puede cargar pesos del checkpoint de AffectNet si se provee ruta.

Este archivo define la lógica de extracción y proyección de características de frames en vectores de embedding.

### `src/dataio.py`

Funciones auxiliares para lectura y escritura de índices:

- `read_sequences_csv(sequences_csv)`: lee CSV de secuencias y valida columnas.
- `parse_paths_joined(paths_joined)`: separa rutas de frames unido por `|`.
- `save_feature_index(records, out_csv)`: guarda un índice de archivos `.npy`.
- `load_feature_index(index_csv)`: carga un CSV de características.

### `src/train.py`

Implementa el proceso de entrenamiento sobre secuencias de features guardadas en `.npy`.

Elementos clave:

- `NpySeqDataset`: dataset que carga arrays de features desde `.npy` y normaliza por feature si se especifica.
- `compute_mean_std(feature_paths)`: calcula media y desviación estándar por dimensión sobre el set de entrenamiento.
- `collate_pad(batch)`: rellena secuencias de longitud variable con padding para formar batches.
- `split_by_persona(...)`: divide el dataset por identidad de sujeto para evitar fuga de información.
- `default_person_split(...)`: particiona personas en conjuntos de entrenamiento, validación y prueba.
- `WarmupScheduler`: scheduler de tasa de aprendizaje con fase de warmup.
- `train_lstm(...)`: función principal de entrenamiento.

Flujo de `train_lstm`:

1. Lee el índice de features y construye etiquetas con `LabelEncoder`.
2. Divide los datos por persona en entrenamiento, validación y prueba.
3. Normaliza features si `normalize=True`.
4. Crea `DataLoader` con padding dinámico.
5. Opcionalmente habilita un `WeightedRandomSampler` para balancear clases en entrenamiento.
6. Elige arquitectura:
   - `lstm`: usa `get_sequence_model('lstm', ...)`.
   - `transformer`: usa `get_sequence_model('transformer', ...)`.
7. Optimiza con `AdamW` y pérdida `CrossEntropyLoss` ponderada por frecuencia de clase.
8. Realiza entrenamiento por épocas con evaluación sobre validación.
9. Guarda checkpoints de cada época y el mejor modelo en `out_dir/checkpoints/best.pt`.
10. Guarda metadatos en `train_meta.json`.
11. Evalúa el mejor modelo en el conjunto de prueba.

Salida esperada del entrenamiento:

- Carpeta `out_dir/checkpoints/` con checkpoints por época y el mejor modelo.
- Archivo `train_meta.json` con:
  - clases de salida.
  - split de personas.
  - mejor checkpoint.
  - historial de pérdidas y exactitud.

## Uso recomendado

### Requisitos mínimos

Las dependencias necesarias incluyen al menos:

- Python 3.10 o superior
- numpy
- pandas
- torch
- torchvision
- scikit-learn
- opencv-python
- dlib
- requests
- ultralytics
- tqdm

### Extraer características con ResNet

Ejecuta el extractor para generar features del dataset actual:

```bash
python scripts/01_extract_resnet.py --sequences_csv outputs_dataset_index/sequences.csv --out_dir pipeline_out_resnet
```

Parámetros importantes:

- `--sequences_csv`: ruta al CSV de secuencias.
- `--out_dir`: carpeta donde se almacenarán `features_resnet` y `features_index_resnet.csv`.
- `--aligned_face_size`: tamaño de la imagen facial que recibirá ResNet.
- `--window_size`: cantidad de frames alrededor del apex.

### Unir datasets

Si existen dos índices de extracción separados para CASME II y SMIC, combina ambos en un dataset maestro:

```bash
python scripts/02_unir_dataset.py
```

El script busca automáticamente:

- `pipeline_out_resnet_casme/features_index_resnet.csv`
- `pipeline_out_resnet_smic/features_index_resnet.csv`

y guarda el resultado en:

- `pipeline_out_resnet_maestro/features_index_maestro.csv`

### Entrenar el modelo

El entrenamiento se realiza desde Python llamando a `src.train.train_lstm`.

No hay un entrypoint CLI directo en `src/train.py` en el estado actual, por lo que es necesario ejecutar un script Python con una llamada similar a:

```python
from src.train import train_lstm
import pandas as pd

index = pd.read_csv('pipeline_out_resnet_maestro/features_index_maestro.csv')
train_lstm(index, out_dir='out_train', epochs=20, batch_size=16, lr=1e-3)
```

### Input esperado para `train_lstm`

El DataFrame debe contener al menos estas columnas:

- `persona`
- `camara`
- `emocion`
- `feature_path`

`feature_path` debe apuntar a archivos `.npy` generados por el extractor con forma `[num_frames, embedding_dim]`.

## Funcionamiento del modelo

### Flujo general

1. Se extraen embeddings por frame con ResNet-18.
2. Cada secuencia de frames se guarda como un `.npy` de dimensión `[T, D]`.
3. El dataset carga estas secuencias y las normaliza si está configurado.
4. Las secuencias se agrupan en batches con padding y longitudes originales.
5. El modelo recurrente procesa cada secuencia completa.
6. Se aplica atención temporal para sacar un vector de contexto.
7. El vector de contexto se usa en un clasificador final para predecir una emoción.

### Arquitectura disponible

- `EmotionLSTM`:
  - LSTM bidireccional configurable.
  - Atención temporal para enfocar frames más relevantes.
  - Clasificador final con dropout para regularizar.
  - Baseline subtraction opcional sobre los primeros frames.

- `FrameEncoderResNet18`:
  - Usa la arquitectura ResNet-18 preentrenada en ImageNet.
  - Elimina la capa final de clasificación y usa `nn.Identity` para obtener embeddings.
  - Aplica una máscara residual ligera antes de la proyección final.

### Entrenamiento

- Optimización con `AdamW`.
- Pérdida `CrossEntropyLoss` con pesos de clase inversos a la frecuencia.
- Scheduler de warmup para transformadores.
- Scheduler `ReduceLROnPlateau` basado en la pérdida de validación.
- Early stopping cuando no hay mejora en validación.

## Observaciones importantes

- El archivo `src/train.py` importa `from .model_transformer import get_sequence_model`, pero actualmente no existe `src/model_transformer.py` en este repositorio. Si se desea usar la opción `arch='transformer'`, es necesario disponer de ese módulo con la función `get_sequence_model`.
- El script `scripts/01_extract_resnet.py` espera que los assets de detección facial se descarguen automáticamente en `assets/models/`.
- El pipeline asume que las rutas de frames en `paths_joined` son accesibles desde el sistema y que están separadas con `|`.

## Resumen del flujo

1. Preparar los CSV de secuencias en `outputs_dataset_index/`.
2. Ejecutar `scripts/01_extract_resnet.py` para generar features.
3. Ejecutar `scripts/02_unir_dataset.py` para consolidar datasets.
4. Usar `src.train.train_lstm` para entrenar y evaluar.

## Archivos clave

- `src/preprocessing.py`: carga de face detector, landmarks y extracción de features por frame.
- `src/model.py`: definición de modelos LSTM y encoder ResNet.
- `src/dataio.py`: utilidades de lectura/escritura de índices.
- `src/train.py`: entrenamiento, validación y test.
- `scripts/01_extract_resnet.py`: extracción inicial de features.
- `scripts/02_unir_dataset.py`: fusión de datasets.

## Recomendaciones

- Verificar que las rutas absolutas en `paths_joined` sean accesibles.
- Mantener organizado el directorio de salida `pipeline_out_resnet` y dataset maestro.
- Comprobar que los modelos de detección y landmarks estén disponibles antes de procesar grandes volúmenes.
- Si se usa GPU, asegurarse de que PyTorch está instalado con soporte CUDA.
