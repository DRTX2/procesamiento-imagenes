from pathlib import Path
import cv2
import numpy as np

# Rutas y configuraciones iniciales
DEFAULT_IMAGE_PATH = Path("informe/formación estelar de W51.png")

CHANNEL_SPECS = {
    "r": {"name": "R", "title": "Canal R - Rojo", "color": "#B46A6A", "index": 0},
    "g": {"name": "G", "title": "Canal G - Verde", "color": "#5F8A63", "index": 1},
    "b": {"name": "B", "title": "Canal B - Azul", "color": "#6E789D", "index": 2},
}

BLOCK_OPTIONS = [2, 4, 8, 16]


def buscar_imagen_inicial():
    # Solo intentamos cargar la imagen de prueba si realmente existe en la carpeta
    return DEFAULT_IMAGE_PATH if DEFAULT_IMAGE_PATH.exists() else None


def cargar_imagen(path=None):
    # Si no nos pasan ruta, devolvemos un canvas negro por defecto
    if not path:
        return np.zeros((320, 480, 3), dtype=np.uint8)
    
    try:
        # Usamos np.fromfile en vez de cv2.imread directo para que no se rompa
        # si la ruta tiene tildes o espacios (un problema clásico en Windows)
        img_array = np.fromfile(str(path), np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("Error al decodificar la imagen")
            
        # OpenCV carga por defecto en BGR, pero necesitamos RGB para que los colores no salgan invertidos
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
    except Exception:
        # En caso de cualquier fallo, mejor devolver el canvas negro antes que crashear el programa
        return np.zeros((320, 480, 3), dtype=np.uint8)


def gris_luma(img_rgb):
    """
    Convierte a escala de grises usando pesos perceptuales (Luma).
    Se ve mucho más natural para el ojo humano que hacer un simple promedio de los 3 canales.
    """
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    gris = 0.299 * r + 0.587 * g + 0.114 * b
    
    return gris.astype(np.uint8)


def expansion_minmax(canal, vmin, vmax):
    # Si los valores están cruzados o son iguales, aplicamos un corte duro
    # para evitar problemas matemáticos (como división por cero).
    if vmax <= vmin:
        return np.where(canal >= vmax, 255, 0).astype(np.uint8)
        
    # Pasamos a float32 para no perder decimales en las operaciones
    f = canal.astype(np.float32)
    
    # Aplicamos la fórmula clásica de ecualización min-max
    out = (f - vmin) * 255.0 / (vmax - vmin)
    
    # El clip asegura que ningún cálculo raro se pase de los límites [0, 255]
    return np.clip(out, 0, 255).astype(np.uint8)


def reducir_resolucion(imagen_gris, n):
    """
    Aplica una compresión espacial dividiendo la imagen en bloques de n x n.
    En lugar de usar dobles bucles for (que matarían el rendimiento en Python), 
    usamos reshape y mean de NumPy para hacerlo de golpe y súper rápido.
    """
    if n <= 1:
        return imagen_gris.copy()

    h, w = imagen_gris.shape
    
    # Recortamos los bordes para que las medidas sean exactamente múltiplos de n
    # Así NumPy no nos lanza errores al hacer el reshape
    hr = (h // n) * n
    wr = (w // n) * n
    recorte = imagen_gris[:hr, :wr].astype(np.float32)
    
    # manejo de vectores, agrupamos en bloques de nxn y sacamos el promedio
    promedios = recorte.reshape(hr // n, n, wr // n, n).mean(axis=(1, 3))
    
    # Ahora volvemos a estirar esos píxeles promediados para que vuelvan a formar un bloque grande
    ampliado = np.repeat(np.repeat(promedios, n, axis=0), n, axis=1)

    # Pegamos el resultado procesado sobre una copia de la imagen original.
    # El truco de esto es que salva los bordes que recortamos al principio,
    # manteniendo intacta la resolución de salida.
    resultado = imagen_gris.copy().astype(np.float32)
    resultado[:hr, :wr] = ampliado
    
    return np.clip(resultado, 0, 255).astype(np.uint8)


def binarizar(imagen_gris, umbral):
    # Blanco puro (255) si supera o iguala el umbral, negro puro (0) si no.
    return np.where(imagen_gris >= umbral, 255, 0).astype(np.uint8)