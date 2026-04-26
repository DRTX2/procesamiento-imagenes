# Herramienta de Preprocesamiento de Imágenes ML

Esta es una interfaz gráfica (GUI) construida con Python (`PySide6`) para realizar tareas de preprocesamiento de imágenes típicamente usadas en Machine Learning y Visión por Computadora.

## Características

1. **Expansión por Min-Max (Contraste):** Normaliza y reescala independientemente los valores (píxeles) en cada canal RGB. También incluye un visor de histogramas en tiempo real.
2. **Compresión (Promedio por Bloques):** Reduce la resolución de la imagen (N x N) promediando sus valores.
3. **Binarización (Umbralización):** Convierte la imagen a puro Blanco y Negro usando un Thresholding o Umbral estricto (0 o 255).

---

## 🚀 Instalación y Despliegue (Linux & Windows)

### Requisitos Previos
- Python 3.9 o superior.

### 1. Clonar o descargar este proyecto

### 2. Crear un entorno virtual (Recomendado)
Esto aislará las dependencias para que no interfieran con el sistema.

**En Windows:**
```cmd
python -m venv env
.\env\Scripts\activate
```

**En Linux / macOS:**
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Instalar Dependencias
Se debe asegurar que el entorno está activo antes de correr.
```bash
pip install -r requirements.txt
```

### 4. Iniciar la Aplicación
```bash
python pro_imagenes.py
```

## Notas adicionales de compatibilidad (Windows)
- **Problema conocido con OpenCV en Windows:** Ocasionalmente las librerías nativas tienen problemas leyendo archivos cuando la ruta tiene caracteres no ASCII (ej. "C:\Users\Juan Pérez\imagen.png"). La aplicación lo previene nativamente cargando flujos de bytes (con `numpy.fromfile`) para asegurar 100% de compatibilidad.
