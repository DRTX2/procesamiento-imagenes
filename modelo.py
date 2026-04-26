from pathlib import Path
import cv2
import numpy as np

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
CHANNEL_SPECS = {
    "r": {"name": "R", "title": "Canal R - Rojo", "color": "#B46A6A", "index": 0},
    "g": {"name": "G", "title": "Canal G - Verde", "color": "#5F8A63", "index": 1},
    "b": {"name": "B", "title": "Canal B - Azul", "color": "#6E789D", "index": 2},
}
BLOCK_OPTIONS = [2, 4, 8, 16]

def buscar_imagen_inicial():
    candidatos = [
        "image.png",
        "image.jpg",
        "image.jpeg",
        "image.bmp",
        "image.tif",
        "image.tiff",
    ]

    for nombre in candidatos:
        ruta = Path(nombre)
        if ruta.exists():
            return ruta

    for patron in IMAGE_EXTENSIONS:
        rutas = sorted(Path(".").glob(patron))
        if rutas:
            return rutas[0]

    return None

def cargar_imagen(path=None):
    if not path:
        return np.zeros((320, 480, 3), dtype=np.uint8)
    try:
        # Cross-platform seguro (especialmente Windows) para rutas con tildes, ñ, etc.
        img_array = np.fromfile(str(path), np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError()
    except Exception:
        return np.zeros((320, 480, 3), dtype=np.uint8)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def gris_luma(img_rgb):
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

def expansion_minmax(canal, vmin, vmax):
    if vmax <= vmin:
        return np.where(canal >= vmax, 255, 0).astype(np.uint8)
    f = canal.astype(np.float32)
    out = (f - float(vmin)) * 255.0 / float(vmax - vmin)
    return np.clip(out, 0, 255).astype(np.uint8)

def reducir_resolucion(imagen_gris, n):
    if n <= 1:
        return imagen_gris.copy()

    h, w = imagen_gris.shape
    hr, wr = (h // n) * n, (w // n) * n
    rec = imagen_gris[:hr, :wr].astype(np.float32)
    prom = rec.reshape(hr // n, n, wr // n, n).mean(axis=(1, 3))
    amp = np.repeat(np.repeat(prom, n, axis=0), n, axis=1)

    res = imagen_gris.copy().astype(np.float32)
    res[:hr, :wr] = amp
    return np.clip(res, 0, 255).astype(np.uint8)

def binarizar(imagen_gris, umbral):
    return np.where(imagen_gris >= umbral, 255, 0).astype(np.uint8)
