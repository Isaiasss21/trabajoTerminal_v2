# Quick Start - Comandos Listos para Ejecutar

## 1) Instalar dependencias (una sola vez)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy pandas scikit-learn tqdm opencv-python ultralytics dlib torch torchvision timm gdown
```

## 2) Descargar dataset desde Drive

```powershell
python .\scripts\00_drive_sync.py --dataset_out_dir .\datasets
```

## 3) Extraer features CASME II

```powershell
python .\scripts\01_extract_resnet.py ^
  --sequences_csv .\outputs_dataset_index\sequences_casme2.csv ^
  --out_dir .\pipeline_out_resnet_casme ^
  --device auto
```

## 4) Extraer features SMIC

```powershell
python .\scripts\01_extract_resnet.py ^
  --sequences_csv .\outputs_dataset_index\sequences_smic.csv ^
  --out_dir .\pipeline_out_resnet_smic ^
  --device auto
```

## 5) Unir datasets

```powershell
python .\scripts\02_unir_dataset.py
```

## 6) Entrenar (LSTM + Atención Temporal)

```powershell
python .\scripts\03_train_base.py ^
  --features_index .\pipeline_out_resnet_maestro\features_index_maestro.csv ^
  --out_dir .\outputs\exp_lstm ^
  --validation_mode loso ^
  --epochs 35 ^
  --batch 8 ^
  --device auto
```

## 7) Entrenar con Transformer

```powershell
python .\scripts\03_train_base.py ^
  --features_index .\pipeline_out_resnet_maestro\features_index_maestro.csv ^
  --out_dir .\outputs\exp_transformer ^
  --arch transformer ^
  --epochs 35 ^
  --batch 8 ^
  --device auto
```

## 8) Entrenar multimodal (Swin + Landmarks)

Primero extrae con landmarks:

```powershell
python .\scripts\01_extract_resnet.py ^
  --sequences_csv .\outputs_dataset_index\sequences_casme2.csv ^
  --out_dir .\pipeline_out_resnet_casme ^
  --save_clips --save_landmarks ^
  --device auto

python .\scripts\01_extract_resnet.py ^
  --sequences_csv .\outputs_dataset_index\sequences_smic.csv ^
  --out_dir .\pipeline_out_resnet_smic ^
  --save_clips --save_landmarks ^
  --device auto
```

Luego unir y entrenar:

```powershell
python .\scripts\02_unir_dataset.py

python .\scripts\03_train_base.py ^
  --features_index .\pipeline_out_resnet_maestro\features_index_maestro.csv ^
  --out_dir .\outputs\exp_multimodal ^
  --input_mode multimodal ^
  --epochs 35 ^
  --batch 8 ^
  --device auto
```

## 9) Fine-tuning desde checkpoint

```powershell
python .\scripts\04_fine_tuning.py ^
  --features_index .\pipeline_out_resnet_maestro\features_index_maestro.csv ^
  --pretrained_weights .\outputs\exp_lstm\checkpoints\best.pt ^
  --out_dir .\outputs\exp_finetune ^
  --epochs 60 ^
  --batch 8 ^
  --device auto
```

## 10) Test rápido (smoke test)

```powershell
python .\scripts\03_train_base.py ^
  --features_index .\pipeline_out_resnet_maestro\features_index_maestro.csv ^
  --out_dir .\outputs\smoke_test ^
  --epochs 2 ^
  --batch 4 ^
  --device auto
```

---

## Resultados en:

- `outputs/exp_lstm/checkpoints/best.pt` → modelo entrenado
- `outputs/exp_lstm/train_meta.json` → métricas y historial
- Consola imprime: `[TEST] acc=... f1_macro=...`
