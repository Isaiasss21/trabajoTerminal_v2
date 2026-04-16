# Micro-Expression Pipeline (LOSO + Multimodal)

Guia rapida para ejecutar extraccion, fusion y entrenamiento en este repositorio.
Esta version esta pensada para pruebas de rendimiento en Windows, incluyendo datasets alojados en Drive.

## 1) Requisitos

- Python 3.10 o 3.11
- GPU NVIDIA opcional (si no, funciona en CPU)
- Espacio en disco suficiente para:
  - features `.npy`
  - clips alineados (si activas multimodal)
  - landmarks (si activas multimodal)
  - checkpoints

Dependencias principales usadas por los scripts:

- `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `tqdm`
- `opencv-python`, `ultralytics`, `dlib`
- `timm` (solo para flujo multimodal/Swin)
- `gdown` (descarga directa desde Google Drive)

## 2) Configuracion de entorno

Desde la raiz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn tqdm opencv-python ultralytics dlib
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm
pip install gdown
```

Si no tienes CUDA, reemplaza la linea de `torch torchvision` por instalacion CPU.

## 3) Dataset en Drive (directo, automatico)

Ahora puedes bajar indices y dataset directo desde Drive sin sincronizar manualmente.
El script ya trae por defecto tu carpeta compartida de Drive (CASME II, SMIC y MEME-Uni).

### 3.1 Sincronizacion automatica

Usa el script nuevo:

```powershell
python .\scripts\00_drive_sync.py ^
  --casme_csv_url "<URL_DRIVE_SEQUENCES_CASME2>" ^
  --smic_csv_url "<URL_DRIVE_SEQUENCES_SMIC>" ^
  --dataset_out_dir .\datasets
```

Notas:

- `--dataset_folder_url` es opcional: por defecto usa
  `https://drive.google.com/drive/folders/1HvrEL14eyqlYacr6MtqjSzCW2phhPf5t?usp=sharing`
- Si en vez de carpeta compartes un archivo (zip/tar), usa `--dataset_file_url`.
- El script reescribe automaticamente `paths_joined` en:
  - `outputs_dataset_index/sequences_casme2.csv`
  - `outputs_dataset_index/sequences_smic.csv`
- Detecta automaticamente el prefijo original en cada CSV (`--replace_from auto`).
- Si no pasas `--replace_to`, intenta enrutar por dataset a subcarpetas descargadas (`casme`, `smic`, `meme`) y, si no encuentra coincidencia, usa `datasets`.

### 3.2 Ajuste de prefijo personalizado (si tu CSV usa otro root)

```powershell
python .\scripts\00_drive_sync.py ^
  --casme_csv_url "<URL_DRIVE_SEQUENCES_CASME2>" ^
  --smic_csv_url "<URL_DRIVE_SEQUENCES_SMIC>" ^
  --replace_from "E:\mis_datos_viejos" ^
  --replace_to "C:\Users\ivonn\trabajoTerminal_v2\datasets\datasetVideosTT"
```

Chequeo minimo antes de extraer:

- `outputs_dataset_index/sequences_casme2.csv`
- `outputs_dataset_index/sequences_smic.csv`

Ambos deben contener `paths_joined` apuntando a archivos existentes en tu maquina.

## 4) Paso a paso (pipeline completo)

### 4.1 Extraer features CASME II

```powershell
python .\scripts\01_extract_resnet.py ^
  --sequences_csv .\outputs_dataset_index\sequences_casme2.csv ^
  --out_dir .\pipeline_out_resnet_casme ^
  --device auto
```

### 4.2 Extraer features SMIC

```powershell
python .\scripts\01_extract_resnet.py ^
  --sequences_csv .\outputs_dataset_index\sequences_smic.csv ^
  --out_dir .\pipeline_out_resnet_smic ^
  --device auto
```

### 4.3 (Opcional) Extraer tambien clips + landmarks para multimodal

Ejecuta los pasos 4.1 y 4.2 agregando:

```text
--save_clips --save_landmarks
```

### 4.4 Unir datasets (CASME + SMIC)

Este script espera por defecto:

- `pipeline_out_resnet_casme/features_index_resnet.csv`
- `pipeline_out_resnet_smic/features_index_resnet.csv`

Comando:

```powershell
python .\scripts\02_unir_dataset.py
```

Salida esperada:

- `pipeline_out_resnet_maestro/features_index_maestro.csv`

### 4.5 Entrenamiento base (LOSO)

```powershell
python .\scripts\03_train_base.py ^
  --features_index .\pipeline_out_resnet_maestro\features_index_maestro.csv ^
  --out_dir .\outputs\exp_base ^
  --validation_mode loso ^
  --input_mode auto ^
  --loss_type ce ^
  --epochs 35 ^
  --batch 8 ^
  --lr 5e-4 ^
  --device auto
```

### 4.6 Entrenamiento base multimodal (Swin + landmarks)

Requiere que en extraccion existan `clip_path` y `landmarks_path` validos.

```powershell
python .\scripts\03_train_base.py ^
  --features_index .\pipeline_out_resnet_maestro\features_index_maestro.csv ^
  --out_dir .\outputs\exp_base_multimodal ^
  --validation_mode loso ^
  --input_mode multimodal ^
  --loss_type focal ^
  --focal_gamma 2.0 ^
  --epochs 35 ^
  --batch 8 ^
  --lr 5e-4 ^
  --device auto
```

### 4.7 Fine-tuning desde pesos previos

```powershell
python .\scripts\04_fine_tuning.py ^
  --features_index .\pipeline_out_resnet_maestro\features_index_maestro.csv ^
  --pretrained_weights .\outputs\exp_base\checkpoints\best.pt ^
  --out_dir .\outputs\exp_finetune ^
  --validation_mode loso ^
  --input_mode auto ^
  --loss_type ce ^
  --epochs 60 ^
  --batch 8 ^
  --lr 1e-4 ^
  --device auto
```

## 5) Salidas y donde mirar rendimiento

Por experimento (ejemplo `outputs/exp_base`):

- `checkpoints/epoch_XXX.pt`
- `checkpoints/best.pt`
- `train_meta.json`

`train_meta.json` incluye:

- clases
- split train/val/test
- historial (`train_loss`, `val_loss`, `val_acc`, `val_f1`)
- `best_checkpoint`

Durante entrenamiento tambien veras en consola:

- `[EP ...] train_loss=... val_loss=... val_acc=... val_f1=...`
- `[TEST] acc=... f1_macro=...`

## 6) Diagnosticos importantes ya integrados

El motor imprime marcadores para detectar problemas tipicos:

- `[CHECK][F1_VS_ACC]` y `[WARN][F1_VS_ACC]`
  - Si Accuracy sube pero F1 macro no, posible desbalance.
- `[WARN][ALIGNMENT_LEAK]`
  - Advierte convergencia sospechosamente rapida (posible atajo por sujeto/alineacion).
- `[CHECK][SWIN_SHAPE]` / `[WARN][SWIN_SHAPE]`
  - Verifica shape del primer clip para Swin.
- `[CHECK][SWIN_BATCH]` / `[WARN][SWIN_BATCH]`
  - Verifica shape real del primer batch multimodal.
- `[CHECK][FOCAL_TREND]`
  - Seguimiento de mejora con Focal Loss.

## 7) Smoke test rapido (para validar setup)

```powershell
python .\scripts\03_train_base.py ^
  --features_index .\pipeline_out_resnet_maestro\features_index_maestro.csv ^
  --out_dir .\outputs\smoke_test ^
  --validation_mode loso ^
  --epochs 2 ^
  --batch 4 ^
  --device auto
```

Si esto corre, tu entorno y rutas estan bien para lanzar experimentos largos.

## 8) Problemas frecuentes

1. `ModuleNotFoundError`:
   - Activa el entorno correcto y reinstala dependencias.
2. Error de rutas en `paths_joined`:
   - Verifica que Drive este montado/sincronizado y accesible.
3. Multimodal falla por clips/landmarks faltantes:
   - Repite extraccion con `--save_clips --save_landmarks`.
4. CUDA no detectada:
   - Usa `--device cpu` o reinstala PyTorch con wheel CUDA compatible.

## 9) Atajos utiles

Entrenar un solo sujeto como test LOSO:

```powershell
python .\scripts\03_train_base.py ^
  --features_index .\pipeline_out_resnet_maestro\features_index_maestro.csv ^
  --out_dir .\outputs\single_subject_test ^
  --validation_mode loso ^
  --test_subject p003 ^
  --epochs 10
```

Con esto puedes iterar rapido antes de lanzar todos los folds.
