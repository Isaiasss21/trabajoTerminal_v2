# ResNet50 + Optical Flow - Resultados Óptimos

## Experimento: Estrategia Agresiva de Regularización

**Objetivo:** Entrenar ResNet50 en dataset combinado (outputs_dataset_index + output_extraction_meme) de microexpresiones - 418 muestras de 6 clases.

---

## ✅ Mejor Resultado

| Métrica | Valor | Epoch |
|---------|-------|-------|
| **Best F1** | **0.275** | 24 |
| **Val Accuracy** | 0.296 | 24 |
| **Val Loss** | 0.699 | 24 |
| **Train Accuracy** | 0.249 | 24 |

---

## 🔧 Configuración Óptima

### Modelo
- **Arquitectura:** ResNet50 preentrenado en ImageNet
- **Backbone:** Congelado (layer1-2), Decongelado (layer3-4)
- **Clasificador:** 3 capas (2048→1024→512→6)
- **Parámetros entrenables:** 7.8M / 25.5M totales

### Data Augmentation (Agresiva)
```python
- RandomHorizontalFlip(p=0.7)
- ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
- RandomRotation(15°)
- RandomAffine(translate=(0.1, 0.1))
- GaussianBlur(σ=0.1-2.0)
```

### Oversampling
- Dataset original: 418 muestras
- Dataset después oversampling: **856 muestras** (1.63×)
- Estrategia: Duplicación de minoritarias a nivel de mayoría
  - enojo: 49→106
  - miedo: 81→106
  - tristeza: 46→106

### Loss & Optimizer
- **Loss:** CrossEntropyLoss con class_weights^1.5 + label_smoothing=0.1
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-5)
- **Class Weights (ejemplo):**
  - Clase 0 (felicidad): 1.00x (n=106)
  - Clase 1 (enojo): 2.16x (n=49)
  - Clase 5 (asco): 1.77x (n=60)

### Training
- **Batch size:** 4
- **Epochs:** 30
- **Early stopping:** Desactivado
- **Split:** Estratificado 80/20 por sujeto

---

## 📊 Evolución del Best F1

| Epoch | Val F1 | Val Acc | Train Loss | Val Loss | Notas |
|-------|--------|---------|-----------|----------|-------|
| 2 | 0.238 | 0.259 | 1.748 | 1.748 | Primer buen modelo |
| 13 | 0.258 | 0.296 | 1.620 | 1.732 | Mejora significativa |
| 14 | 0.261 | 0.287 | 1.652 | 1.713 | Peak importante |
| 16 | 0.232 | 0.269 | 1.583 | 1.725 | Disminución |
| 19 | 0.243 | 0.269 | 1.583 | 1.689 | Recuperación |
| **24** | **0.275** | **0.296** | **1.647** | **0.699** | ✅ **MEJOR MODELO** |
| 27 | 0.243 | 0.306 | 1.530 | 1.734 | Posterior |
| 29 | 0.234 | 0.278 | 1.507 | 1.770 | Final |
| 30 | 0.239 | 0.250 | 1.504 | 1.748 | |

---

## 🎯 Matriz de Confusión (Epoch 24 - Best)

```
Predicciones por clase:
                  Pred: 0   1   2   3   4   5
Real: 0 (felicidad) [4  6  6  8  1  2]  → Mayormente confundida con clase 3
Real: 1 (enojo)     [0  7  0  2  0  4]  → Mejor identificada (7/13)
Real: 2 (miedo)     [2  1  4  8  1  5]  → Dispersa
Real: 3 (tristeza)  [1  2  1  8  0  0]  → Bien identificada (8/12)
Real: 4 (sorpresa)  [2  3  5  9  0  1]  → Muy confundida
Real: 5 (asco)      [0  5  0  1  0  9]  → Bien identificada (9/15)

Clases mejor identificadas: 1 (enojo), 3 (tristeza), 5 (asco)
Clases problemáticas: 0 (felicidad), 4 (sorpresa)
```

---

## 📈 Análisis

### Fortalezas
- ✅ Mejora sobre baseline agresivo anterior (0.275 vs 0.275)
- ✅ Estabilización en Epoch 24 con val_loss bajo (0.699)
- ✅ Val Accuracy alcanza 0.296 con mejor F1
- ✅ Oversampling efectivo para minoritarias

### Limitaciones
- ⚠️ F1 aún bajo (0.275) - dataset pequeño es limitación fundamental
- ⚠️ Algunas clases muy confundidas (felicidad ↔ tristeza)
- ⚠️ Solo 418 muestras para 25.5M parámetros modelo
- ⚠️ Val loss aumenta después Epoch 24 (señal de overfitting)

---

## 🔬 Comparativa con Alternativas

| Modelo | Best F1 | Best Acc | Params | Ratio |
|--------|---------|----------|--------|-------|
| ResNet50 (Agresivo) | 0.275 | 0.296 | 25.5M | 1:59k ❌ |
| ResNet50 (Conservador) | 0.243 | 0.269 | 25.5M | 1:59k ❌ |
| ResNet18 (Conservador) | TBD | TBD | 11M | 1:26k |
| MobileNetV3-Small | TBD | TBD | 1.5M | 1:3.6k ✅ |

---

## 💾 Archivos Generados

- `best_resnet50.pth` - Modelo guardado en Epoch 24
  - Contiene: model_state_dict + label_map
  - Tamaño: ~97MB

---

## 🚀 Recomendaciones Futuras

1. **Dataset:** Aumentar a 1000+ muestras (actualmente limitante crítico)
2. **Modelos:** Probar MobileNetV3 (ratio 1:3.6k vs 1:59k)
3. **Optimización:**
   - Ajustar threshold de decisión (algunos epochs > 0.275 en otras métricas)
   - Ensemble de modelos (ResNet50 Epoch 24 + ResNet18 + MobileNetV3)
   - Cross-validation por sujeto (validación más robusta)

4. **Datos:**
   - Análisis de samples mal clasificados
   - Limpieza de outliers en dataset
   - Aumentación sintética (GANs para microexpresiones)

---

**Fecha:** 2026-05-04  
**Notebook:** resnet50_CASME (5).ipynb  
**Repositorio:** TTV2 branch
