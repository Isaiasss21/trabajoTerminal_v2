# Documentación de la Interfaz Gráfica de Usuario
## Sistema de Análisis de Microexpresiones Faciales

---

## 1. Descripción General

La interfaz gráfica del sistema fue desarrollada con el framework **PyQt6**, siguiendo un diseño de tema oscuro (*dark theme*) con una paleta de colores consistente. La ventana principal adopta un patrón de navegación lateral (*sidebar navigation*), en el que un panel fijo en el costado izquierdo contiene los botones de navegación entre pantallas, mientras que el área derecha actúa como un contenedor de vistas que intercambia el contenido activo sin recargar la ventana.

La aplicación se organiza en **cuatro pantallas principales**:

| Pantalla | Función |
|---|---|
| Análisis | Carga de video, procesamiento y visualización en tiempo de análisis |
| Resultados | Resumen estadístico y gráfica de la sesión analizada |
| Historial | Lista de todas las sesiones almacenadas con acciones sobre cada una |
| Ajustes | Configuración del modelo de inferencia, cámara y preferencias |

---

## 2. Estructura General de la Ventana

La ventana principal (`MainWindow`) está construida sobre `QMainWindow` y se divide horizontalmente en dos zonas:

```
┌──────────────┬────────────────────────────────────────────────┐
│   Sidebar    │          Área de contenido (pantalla activa)   │
│   200 px     │                                                │
│              │                                                │
│  ▶ Análisis  │   La pantalla seleccionada ocupa todo          │
│  📊 Resultados│   este espacio y cambia sin recargar          │
│  🗂 Historial │   la ventana.                                  │
│  ⚙ Ajustes   │                                                │
│              │                                                │
│  [Versión]   │                                                │
└──────────────┴────────────────────────────────────────────────┘
```

El **sidebar** contiene botones de navegación que indican visualmente cuál es la pantalla activa mediante un borde izquierdo de color de acento (`#2979FF`) y un fondo semitransparente. Al hacer clic en cualquier botón, el contenido del área principal se reemplaza instantáneamente.

---

## 3. Pantalla de Análisis

### 3.1 Descripción

La pantalla de análisis es el componente central del sistema. Permite al usuario cargar un video pre-grabado y ejecutar el pipeline completo de detección facial, extracción de flujo óptico e inferencia de microexpresiones. El diseño está organizado en tres zonas funcionales: barra de control superior, zona de visualización de video central y panel de resultados lateral derecho.

### 3.2 Distribución Visual

```
┌──────────────────────────────────────────────────────────────────────┐
│  [📂 Seleccionar video]  nombre_video.mp4  Secuencias: 8  [▶ Analizar] [■ Cancelar] │
├────────────────────────────────────────────────────────────────┬──────────────────┤
│                                                                │ Última predicción│
│  ┌──────────────────────┐   ┌──────────────────────┐          │                  │
│  │  Video original      │   │  Landmarks / ROI     │          │ Neutral          │
│  │                      │   │                      │          │ Confianza: 78.5% │
│  │  [frame del video]   │   │  [frame anotado con  │          │ ✔ Detección      │
│  │                      │   │   landmarks y bbox]  │          │   válida         │
│  └──────────────────────┘   └──────────────────────┘          │ ─────────────── │
│                                                                │ Distribución     │
│  ████████████████████░░░░  Frame 240 / 450                    │ Alegría  ▓░  7%  │
└────────────────────────────────────────────────────────────────┴──────────────────┘
```

### 3.3 Componentes de la Interfaz

#### Barra de control superior

La barra superior concentra todos los controles de inicio y cancelación del análisis:

- **Botón "📂 Seleccionar video"**: Abre un diálogo de selección de archivos filtrado por formatos de video comunes (`.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`). Una vez seleccionado, muestra el nombre del archivo junto al botón.
- **Etiqueta del archivo**: Muestra el nombre del video seleccionado o el texto *"Ningún video seleccionado"* en su estado inicial.
- **Contador de secuencias**: Muestra en tiempo real cuántas secuencias completas de flujo óptico han sido procesadas e inferidas durante el análisis activo.
- **Botón "▶ Analizar"**: Inicia el pipeline de procesamiento. Permanece deshabilitado hasta que se seleccione un video y el modelo esté cargado. Al activarse, cambia a color verde como indicador visual.
- **Botón "■ Cancelar"**: Interrumpe el análisis en curso. Solo se activa durante el procesamiento activo.

#### Zona de visualización dual

El centro de la pantalla muestra dos paneles de video en paralelo, con igual proporción de espacio:

- **Panel izquierdo — "Video original"**: Muestra el frame del video tal como fue capturado, sin ningún procesamiento adicional. Permite al evaluador contrastar la imagen limpia con la versión anotada.

- **Panel derecho — "Landmarks / ROI"**: Muestra el mismo frame con las siguientes anotaciones visuales superpuestas:
  - **Cuadro delimitador de la cara** (*bounding box*): Rectángulo verde de 4 píxeles de grosor que enmarca la región facial detectada por MediaPipe.
  - **Landmarks generales**: Puntos cian (radio 2 px) distribuidos sobre los 468 puntos de la malla facial de MediaPipe, que representan la geometría completa del rostro.
  - **Landmarks de regiones de interés**: Puntos de mayor tamaño (radio 5 px) con colores diferenciados por región:
    - Naranja: cejas izquierda y derecha
    - Azul claro: ojos izquierdo y derecho
    - Azul oscuro: boca
    - Verde amarillo: nariz

  Esta visualización permite verificar que la detección facial está operando correctamente y que las regiones de mayor importancia para el model están siendo identificadas.

Los frames se emiten cada 5 frames procesados para no saturar el hilo de interfaz y se escalan proporcionalmente al tamaño disponible del panel.

#### Panel de predicción (lateral derecho)

El panel lateral de ancho fijo (280 px) muestra los resultados de la última secuencia inferida:

- **Emoción predicha**: Nombre de la emoción con tipografía grande (28 px) y color representativo asignado a cada clase.
- **Nivel de confianza**: Porcentaje de confianza de la predicción con un decimal.
- **Indicador de validez**: Texto de estado con código de color:
  - ✔ Detección válida — verde
  - ⚠ Detección incierta — ámbar
  - ✘ Baja confianza — rojo
- **Distribución de probabilidades**: Una barra de progreso por cada clase emocional, con el porcentaje exacto visible a la derecha de cada barra. Las barras se actualizan con cada nueva predicción.

Las clases disponibles son: **Asco, Felicidad, Neutral, Sorpresa ** y **Tristeza**.

#### Barra de progreso

Una barra de progreso lineal en la parte inferior indica el avance del procesamiento en porcentaje, acompañada de una etiqueta de texto con el número de frame actual y el total (e.g., *"Procesando frame 240 / 450"*). Al finalizar, la etiqueta cambia a *"✔ Análisis completado — N secuencias procesadas"* en color verde.

### 3.4 Flujo de Operación

El flujo de operación de esta pantalla sigue la siguiente secuencia:

1. El usuario selecciona un archivo de video.
2. El sistema habilita el botón de análisis.
3. Al iniciar, se crea una nueva sesión en el gestor de almacenamiento y se lanza el hilo `VideoPipeline`.
4. El hilo emite frames de previsualización, progreso, resultados de secuencias y señal de finalización.
5. Cada resultado de secuencia se muestra en el panel lateral y se guarda en la sesión activa.
6. Al concluir, la sesión se cierra automáticamente y la pantalla emite la señal `session_finished` con el identificador único de sesión, lo que provoca la navegación automática a la pantalla de Resultados.

---

## 4. Pantalla de Resultados

### 4.1 Descripción

La pantalla de resultados presenta un resumen estadístico completo de la sesión de análisis más reciente. Se accede automáticamente al finalizar un análisis, aunque también puede consultarse desde el historial.

### 4.2 Componentes de la Interfaz

#### Encabezado

Contiene el título *"Resultados — [session_id]"* y el botón **"⬇ Exportar CSV"** que se habilita cuando hay datos disponibles.

#### Tarjeta de métricas resumidas

Presenta en formato de grilla 3×2 los siguientes indicadores:

| Métrica | Descripción |
|---|---|
| Predicciones totales | Número total de secuencias procesadas |
| Predicciones válidas | Secuencias con confianza por encima del umbral mínimo |
| Descartadas | Secuencias de baja confianza excluidas del análisis |
| Emoción dominante | Clase con mayor frecuencia en las predicciones válidas |
| Confianza promedio | Media de los valores de confianza de todas las predicciones |
| Calidad de landmarks | Porcentaje de frames en que MediaPipe detectó correctamente el rostro |

#### Gráfica de distribución

Una gráfica de barras verticales generada con **Matplotlib** integrado en la interfaz PyQt6 muestra la distribución relativa de cada emoción. La gráfica utiliza el tema oscuro consistente con el resto de la aplicación, con barras de color diferenciado por emoción. Solo se muestra cuando hay predicciones válidas registradas.

#### Exportación CSV

El botón de exportación abre un diálogo de guardado del sistema operativo. El archivo generado contiene una fila por predicción con los campos: `session_id`, `timestamp`, `emotion`, `confidence`, `is_valid`, `duration_ms` y `landmarks_detected`.

---

## 5. Pantalla de Historial

### 5.1 Descripción

La pantalla de historial presenta un registro tabular de todas las sesiones de análisis almacenadas en el sistema, con un máximo de 50 sesiones. Permite revisar, exportar y eliminar sesiones anteriores.

### 5.2 Componentes de la Interfaz

#### Tabla de sesiones

Una tabla con las siguientes columnas:

| Columna | Contenido |
|---|---|
| Fecha / Hora | Marca temporal de inicio de la sesión (formato `YYYY-MM-DD HH:MM:SS`) |
| Predicciones | Total de secuencias procesadas en esa sesión |
| Válidas | Predicciones que superaron el umbral de confianza |
| Emoción dominante | Emoción más frecuente en predicciones válidas |
| Confianza media | Promedio de confianza de la sesión |
| Duración (s) | Tiempo total del análisis en segundos |

La tabla permite seleccionar una sola fila a la vez (modo de selección por filas). Las sesiones se ordenan de la más reciente a la más antigua.

#### Barra de acciones

Con una sesión seleccionada en la tabla, se habilitan tres acciones:

- **"👁 Ver resultados"**: Navega a la pantalla de Resultados cargando los datos de la sesión seleccionada.
- **"⬇ Exportar CSV"**: Abre el diálogo de guardado para exportar las predicciones de la sesión seleccionada.
- **"🗑 Eliminar"**: Solicita confirmación y elimina permanentemente la sesión del disco.

El botón **"↺ Actualizar"** recarga la lista desde disco sin necesidad de reiniciar la aplicación.

---

## 6. Pantalla de Ajustes

### 6.1 Descripción

La pantalla de ajustes permite configurar los parámetros operativos de la aplicación, organizados en cuatro secciones mediante `QGroupBox`.

### 6.2 Secciones

#### Modelo de inferencia

Permite cambiar el archivo de pesos del modelo en tiempo de ejecución sin necesidad de reiniciar la aplicación:

- **Campo de ruta**: Muestra la ruta del modelo actualmente cargado. Es de solo lectura y se rellena automáticamente al inicio si el modelo está disponible.
- **Botón "📂 Examinar…"**: Abre un diálogo de selección de archivos filtrado por extensiones `.pth` y `.pt`. El diálogo se abre por defecto en la carpeta `models/` del proyecto.
- **Botón "Cargar modelo"**: Se habilita tras seleccionar un archivo. Invoca al motor de inferencia para cargar los nuevos pesos y actualiza el estado en todas las pantallas.
- **Indicador de estado**: Texto de estado que informa sobre el resultado de la carga (e.g., *"Modelo cargado — 5 clases"* o mensaje de error si falla).

#### Cámara

Permite seleccionar el índice de cámara a utilizar en caso de que el equipo disponga de múltiples dispositivos de captura. El botón **"🔍 Detectar cámaras"** escanea los índices disponibles y puebla el menú desplegable.

#### Consentimiento

Contiene un *checkbox* de consentimiento informado que el usuario debe activar para habilitar el almacenamiento de datos de la sesión, en cumplimiento con los principios de protección de datos personales.

#### Acerca de

Muestra información sobre la versión de la aplicación, el modelo utilizado y los créditos del proyecto.

---

## 7. Consideraciones de Diseño

### 7.1 Tema Visual

La interfaz adopta un esquema de color oscuro (*dark mode*) con la siguiente paleta base:

| Token | Valor | Uso |
|---|---|---|
| `_BG` | `#121212` | Fondo principal de todas las pantallas |
| `_SURFACE` | `#1E1E1E` | Fondo de tarjetas y paneles |
| `_ACCENT` | `#2979FF` | Botones primarios, barra activa del sidebar |
| `_GREEN` | `#4CAF50` | Predicciones válidas, análisis completado |
| `_AMBER` | `#FFC107` | Advertencias, predicciones inciertas |
| `_RED` | `#F44336` | Errores, baja confianza, botón cancelar |
| `_TEXT` | `#E0E0E0` | Texto principal |
| `_SUBTEXT` | `#9E9E9E` | Etiquetas secundarias y descriptivas |

### 7.2 Concurrencia y Respuesta de la Interfaz

El procesamiento de video se ejecuta en un hilo separado (`VideoPipeline`, subclase de `QThread`) para mantener la interfaz completamente responsiva durante el análisis. La comunicación entre el hilo de procesamiento y el hilo principal de la interfaz se realiza exclusivamente mediante señales Qt (`pyqtSignal`), lo que garantiza la seguridad en el acceso a los widgets sin bloqueos ni condiciones de carrera.

### 7.3 Arquitectura de Componentes

```
MainWindow
│
├── Sidebar (navegación)
│   ├── BtnAnálisis
│   ├── BtnResultados
│   ├── BtnHistorial
│   └── BtnAjustes
│
└── QStackedWidget
    ├── AnalysisScreen
    │   ├── Barra de control (selección y botones)
    │   ├── Panel video original (QLabel)
    │   ├── Panel video anotado (QLabel)
    │   └── Panel de predicción (barras + emoción)
    ├── ResultsScreen
    │   ├── Tarjeta de métricas
    │   └── Gráfica matplotlib
    ├── HistoryScreen
    │   ├── Tabla de sesiones (QTableWidget)
    │   └── Barra de acciones
    └── SettingsScreen
        ├── Grupo: Modelo
        ├── Grupo: Cámara
        ├── Grupo: Consentimiento
        └── Grupo: Acerca de
```

---

## 8. Tecnologías Utilizadas en la Interfaz

| Componente | Tecnología |
|---|---|
| Framework de interfaz | PyQt6 (Python bindings para Qt 6) |
| Procesamiento de video | OpenCV (`cv2`) |
| Detección facial y landmarks | MediaPipe Face Landmarker |
| Gráficas estadísticas | Matplotlib con backend `QtAgg` |
| Hilos de procesamiento | `QThread` de Qt |
| Comunicación inter-hilo | Señales y slots de Qt (`pyqtSignal`, `pyqtSlot`) |
