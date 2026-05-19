# MicroExpression Analyzer — Guía de ejecución

Aplicación de escritorio para análisis de microexpresiones faciales en tiempo real usando flujo óptico y redes neuronales profundas.

---

## Requisitos previos

| Requisito | Versión mínima |
|-----------|----------------|
| Python    | 3.11           |
| Webcam    | Cualquier cámara USB o integrada |
| RAM       | 4 GB mínimo (8 GB recomendado) |
| GPU       | Opcional (la app corre en CPU sin problema) |

> **Sistema operativo:** Windows 10/11 (también funciona en Linux/macOS con Python 3.11+)

---

## 1. Instalación

### 1.1 Clonar o copiar el proyecto

Asegúrate de tener la carpeta `microexpression-system/` con la siguiente estructura mínima:

```
microexpression-system/
├── app/
├── models/
│   ├── face_landmarker.task       ← ya incluido
│   └── best_resnet18_flow.pth     ← debes copiarlo aquí (ver paso 1.2)
├── requirements.txt
└── README.md
```

### 1.2 Colocar el modelo entrenado

Copia el archivo `best_resnet18_flow.pth` dentro de la carpeta `models/`:

```
microexpression-system/
└── models/
    └── best_resnet18_flow.pth   ✔
```

> Este archivo contiene los pesos del clasificador de microexpresiones entrenado con ResNet-18 + flujo óptico. Sin él, la app no realizará predicciones.

### 1.3 Crear entorno virtual (recomendado)

Abre una terminal en la carpeta del proyecto:

```powershell
# Entrar a la carpeta del proyecto
cd microexpression-system

# Crear entorno virtual
python -m venv .venv

# Activar el entorno (Windows)
.venv\Scripts\Activate.ps1

# En Linux/macOS
# source .venv/bin/activate
```

### 1.4 Instalar dependencias

```powershell
pip install -r requirements.txt
```

Las dependencias instalan automáticamente:
- `PyQt6` — interfaz gráfica
- `torch` + `torchvision` — inferencia del modelo
- `opencv-python` — captura de cámara y flujo óptico
- `mediapipe` — detección de landmarks faciales
- `numpy`, `Pillow`, `matplotlib`, `pandas`

> **Nota PyTorch CPU vs GPU:** El comando anterior instala PyTorch en versión CPU.  
> Si tienes GPU NVIDIA y quieres aceleración, instala PyTorch con CUDA siguiendo la [guía oficial de PyTorch](https://pytorch.org/get-started/locally/).

---

## 2. Ejecutar la aplicación

Desde la carpeta `microexpression-system/` con el entorno virtual activado:

```powershell
python -m app.main --model models/best_resnet18_flow.pth
```

La ventana de la aplicación se abrirá en unos segundos.

### Opciones adicionales

```powershell
# Forzar uso de CPU (aunque haya GPU disponible)
python -m app.main --model models/best_resnet18_flow.pth --device cpu

# Forzar uso de GPU CUDA
python -m app.main --model models/best_resnet18_flow.pth --device cuda
```

---

## 3. Uso de la interfaz

La aplicación tiene 4 pantallas accesibles desde el panel lateral izquierdo.

### ▶ Análisis (pantalla principal)

1. Conecta tu webcam antes de abrir la app.
2. Haz clic en **"▶ Iniciar"** para comenzar la sesión.
3. Coloca tu rostro frente a la cámara — aparecerá el indicador **"⬤ Cara detectada"** en verde.
4. El panel derecho muestra en tiempo real:
   - **Emoción detectada** (Alegría, Asco, Enojo, Miedo, Neutral, Sorpresa, Tristeza)
   - **Confianza** de la predicción (%)
   - **Barras de distribución** con la probabilidad de cada emoción
   - Indicador de validez (✔ válida / ⚠ incierta / ✘ baja confianza)
5. Haz clic en **"■ Detener"** para finalizar la sesión.
   - La app navega automáticamente a la pantalla de **Resultados**.

> **¿Por qué siempre aparece "Neutral"?**  
> En reposo, sin microexpresiones activas, el modelo correctamente predice "Neutral". Para obtener predicciones de otras emociones, realiza expresiones faciales visibles aunque sean breves (cejas levantadas, fruncir el ceño, etc.).

### 📊 Resultados

Muestra el resumen estadístico de la sesión más reciente:
- Total de predicciones y cuántas fueron válidas
- Emoción dominante
- Confianza promedio
- Gráfica de distribución de emociones
- Botón **"⬇ Exportar CSV"** → abre un diálogo para guardar el archivo donde quieras

### 🗂 Historial

Lista todas las sesiones guardadas (máximo 50). Para cada sesión puedes:
- **👁 Ver resultados** → navegar a la pantalla de Resultados de esa sesión
- **⬇ Exportar CSV** → guardar los datos como archivo CSV
- **🗑 Eliminar** → borrar la sesión del disco

> Para exportar: selecciona una fila de la tabla (queda resaltada) y luego haz clic en **"⬇ Exportar CSV"**.

### ⚙ Ajustes

- Cambiar el modelo cargado (cargar otro archivo `.pth`)
- Seleccionar la cámara a usar si hay varias conectadas

---

## 4. Formato del CSV exportado

Cada fila representa una predicción individual durante la sesión:

| Campo              | Descripción                                      |
|--------------------|--------------------------------------------------|
| `session_id`       | Identificador único de la sesión                 |
| `timestamp`        | Fecha y hora de la predicción (ISO-8601 UTC)     |
| `emotion`          | Emoción predicha en español                      |
| `confidence`       | Confianza de la predicción (0.0 – 1.0)           |
| `intensity`        | Intensidad estimada                              |
| `duration_ms`      | Duración estimada de la microexpresión (ms)      |
| `landmarks_detected` | 1 si MediaPipe detectó el rostro, 0 si no     |

---

## 5. Solución de problemas frecuentes

### La app no encuentra la cámara
- Verifica que la webcam esté conectada **antes** de iniciar la sesión.
- Si tienes varias cámaras, ve a **Ajustes** y selecciona el índice correcto.
- En Windows, verifica que ninguna otra app esté usando la cámara (Teams, Zoom, etc.).

### No aparecen predicciones / botón Iniciar no hace nada
- Probablemente el modelo no está cargado. Verifica que `models/best_resnet18_flow.pth` exista.
- Desde **Ajustes** puedes cargar el modelo manualmente si está en otra ubicación.

### Error al instalar `mediapipe`
```powershell
pip install mediapipe --upgrade
```

### Error al instalar `PyQt6` en Linux
```bash
sudo apt install python3-pyqt6  # Ubuntu/Debian
# o mediante pip:
pip install PyQt6
```

### La ventana se abre pero el video no aparece
- El indicador dice **"Sin cara"** — el modelo de landmarks no detecta rostro. Asegúrate de tener buena iluminación y de estar frente a la cámara.
- Revisa que `models/face_landmarker.task` exista en la carpeta `models/`.

---

## 6. Estructura del proyecto

```
microexpression-system/
├── app/
│   ├── main.py                    ← punto de entrada
│   ├── inference/
│   │   ├── inference_engine.py    ← carga del modelo y predicciones
│   │   └── camera_pipeline.py     ← captura de cámara y flujo óptico
│   ├── storage/
│   │   └── session_manager.py     ← guardado de sesiones
│   ├── analytics/
│   │   └── stats_engine.py        ← cálculo de estadísticas
│   └── ui/
│       ├── main_window.py         ← ventana principal
│       ├── analysis_screen.py     ← pantalla de análisis
│       ├── results_screen.py      ← pantalla de resultados
│       ├── history_screen.py      ← pantalla de historial
│       └── settings_screen.py     ← pantalla de ajustes
├── models/
│   ├── face_landmarker.task       ← modelo MediaPipe (incluido)
│   └── best_resnet18_flow.pth     ← modelo entrenado (proveer manualmente)
├── data/                          ← creado automáticamente al ejecutar
│   ├── sessions/                  ← sesiones guardadas
│   └── exports/                   ← CSVs exportados
└── requirements.txt
```

---

## Referencia rápida

```powershell
# 1. Entrar a la carpeta
cd microexpression-system

# 2. Activar entorno virtual
.venv\Scripts\Activate.ps1

# 3. (Solo primera vez) Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
python -m app.main --model models/best_resnet18_flow.pth
```
