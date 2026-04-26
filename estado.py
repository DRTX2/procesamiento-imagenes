import numpy as np
import modelo

class EstadoPreprocesamiento:
    def __init__(self):
        self.img = np.zeros((320, 480, 3), dtype=np.uint8)
        self.rgb_norm = self.img.copy()
        self.gris_actual = modelo.gris_luma(self.img)
        self.comp_actual = self.gris_actual.copy()
        self.bin_actual = self.gris_actual.copy()
        
        self.image_path = None
        self.rangos = {
            "r": (0, 255),
            "g": (0, 255),
            "b": (0, 255)
        }
        self.bloque = 2
        self.umbral = 128
        
        # Cache para evitar re-dibujar histogramas si el rango no cambia
        self.last_ranges = {
            "r": None,
            "g": None,
            "b": None
        }

    def establecer_imagen(self, ruta):
        self.image_path = ruta
        self.img = modelo.cargar_imagen(ruta)
        self.rgb_norm = self.img.copy()
        self.last_ranges = {"r": None, "g": None, "b": None}
        self.actualizar_normalizacion()

    def actualizar_rango(self, canal, mn, mx):
        self.rangos[canal] = (mn, mx)

    def actualizar_normalizacion(self):
        normalizados = {}
        for key in ["r", "g", "b"]:
            mn, mx = self.rangos[key]
            canal = self.img[:, :, modelo.CHANNEL_SPECS[key]["index"]]
            norm = modelo.expansion_minmax(canal, mn, mx)
            normalizados[key] = norm

        self.rgb_norm = np.stack(
            [normalizados["r"], normalizados["g"], normalizados["b"]], axis=2
        )
        self.actualizar_compresion()

    def actualizar_compresion(self, bloque=None, umbral=None):
        if bloque is not None:
            self.bloque = bloque
        if umbral is not None:
            self.umbral = umbral
            
        self.gris_actual = modelo.gris_luma(self.rgb_norm)
        self.comp_actual = modelo.reducir_resolucion(self.gris_actual, self.bloque)
        self.bin_actual = modelo.binarizar(self.comp_actual, self.umbral)

    def obtener_canal_original(self, key):
        idx = modelo.CHANNEL_SPECS[key]["index"]
        return self.img[:, :, idx]

    def obtener_canal_normalizado(self, key):
        idx = modelo.CHANNEL_SPECS[key]["index"]
        return self.rgb_norm[:, :, idx]
