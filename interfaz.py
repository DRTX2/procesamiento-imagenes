import sys
from pathlib import Path
import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
import modelo

def np_to_pixmap(img_np):
    if img_np.ndim == 2:
        img_np = np.ascontiguousarray(img_np)
        h, w = img_np.shape
        qimg = QImage(img_np.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())

    img_np = np.ascontiguousarray(img_np)
    h, w, _ = img_np.shape
    qimg = QImage(img_np.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def formato_resolucion(imagen):
    alto, ancho = imagen.shape[:2]
    return f"{ancho} x {alto} px"

def estilo_slider(color):
    return f"""
    QSlider::groove:horizontal {{
        border: 1px solid #D7CCBD;
        height: 7px;
        background: #EDE4D9;
        border-radius: 4px;
    }}
    QSlider::sub-page:horizontal {{
        background: {color};
        border-radius: 4px;
    }}
    QSlider::add-page:horizontal {{
        background: #E1D7CB;
        border-radius: 4px;
    }}
    QSlider::handle:horizontal {{
        background: #FFFDF8;
        border: 2px solid {color};
        width: 14px;
        margin: -5px 0;
        border-radius: 9px;
    }}
    """

APP_STYLE = """
QMainWindow { background: #F5EFE6; }
QWidget#AppRoot, QWidget#PageBg, QStackedWidget, QStackedWidget > QWidget { background: #F5EFE6; }
QWidget { color: #3F3A36; font-size: 12px; }
QScrollArea { border: none; background: #F5EFE6; }
QWidget#qt_scrollarea_viewport { background: #F5EFE6; }
QFrame#Card { background: #FFFDF9; border: 1px solid #E0D5C7; border-radius: 14px; }
QFrame#PreviewPanel { background: #FBF7F0; border: 1px solid #E3D9CD; border-radius: 10px; }
QLabel#MainTitle { font-size: 30px; font-weight: 800; color: #3F3A36; }
QLabel#SectionTitle { font-size: 14px; font-weight: 800; color: #4A443F; }
QLabel#Muted { color: #7A6F64; font-size: 12px; }
QLabel#MiniTitle { color: #8A7F73; font-size: 11px; font-weight: 800; letter-spacing: 0.5px; }
QLabel#AccentGreen { color: #5F8A63; font-size: 14px; font-weight: 800; }
QPushButton { background: #5E5A56; color: white; border: none; border-radius: 8px; padding: 10px 14px; font-size: 13px; font-weight: 700; }
QPushButton:hover { background: #4F4B47; }
QPushButton:pressed { background: #403C38; }
QPushButton#PrimaryButton { background: #6E7B57; }
QPushButton#PrimaryButton:hover { background: #5F6D4A; }
QPushButton#SuccessButton { background: #7D9A67; }
QPushButton#SuccessButton:hover { background: #708B5D; }
QPushButton#GhostButton { background: #EFE7DC; color: #5E554C; }
QPushButton#GhostButton:hover { background: #E4DACD; }
QPushButton#NavButton { background: transparent; color: #7A6F64; border: 1px solid #DCCDBD; padding: 8px 16px; }
QPushButton#NavButton[active="true"] { background: #FFFDF9; color: #4A443F; border: 1px solid #CDBCA9; }
QRadioButton { spacing: 8px; color: #4E473F; font-weight: 600; }
QRadioButton::indicator { width: 14px; height: 14px; }
QRadioButton::indicator:unchecked { border: 2px solid #5E554C; border-radius: 9px; background: #FFFDF8; }
QRadioButton::indicator:checked { border: 2px solid #6E7B57; border-radius: 9px; background: #6E7B57; }
"""

class HistCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(3.6, 3.2))
        self.figure.patch.set_facecolor("white")
        super().__init__(self.figure)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.axes = self.figure.subplots(2, 1)
        self.figure.subplots_adjust(left=0.14, right=0.96, top=0.92, bottom=0.12, hspace=0.55)

    def draw_histograms(self, original, normalized, color):
        labels = ("ORIGINAL", "NORMALIZADO")
        for ax, data, label in zip(self.axes, (original, normalized), labels):
            ax.clear()
            hist = np.bincount(data.flatten(), minlength=256)
            ax.bar(np.arange(256), hist, width=1.0, align="edge", color=color, alpha=0.85, edgecolor=color, linewidth=0)
            ax.set_xlim(-1, 256)
            if hist.size > 2:
                ymax = int(hist[1:-1].max())
            else:
                ymax = int(hist.max()) if hist.size else 0
            ax.set_ylim(0, max(1, int(ymax * 1.15)))
            ax.set_facecolor("#FBF7F0")
            ax.set_title(label, fontsize=8, fontweight="bold", color="#7C6F62", pad=10)
            ax.tick_params(labelsize=7, colors="#7D7266")
            ax.grid(alpha=0.12, color="#9C9084", linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#DED3C6")
            ax.spines["bottom"].set_color("#DED3C6")
        self.draw_idle()

class VentanaPrincipal(QMainWindow):
    def __init__(self, estado):
        super().__init__()
        self.estado = estado
        self.setWindowTitle("Ecualizador de Imágenes — Preprocesamiento ML")
        self.resize(1680, 980)
        self.setStyleSheet(APP_STYLE)

        self.channel_widgets = {}
        self.block_buttons = {}
        self.t_slider = None
        self.block_group = None
        self.ui_ready = False

        self._build_ui()
        self.ui_ready = True
        self.actualizar_vistas_normalizacion()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("AppRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 16, 18, 20)
        root_layout.setSpacing(14)

        titulo = QLabel("Preprocesamiento ML de Imágenes")
        titulo.setObjectName("MainTitle")
        titulo.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(titulo)

        nav_row = QHBoxLayout()
        nav_row.addStretch(1)
        self.btn_nav_norm = QPushButton("Normalización")
        self.btn_nav_comp = QPushButton("Compresión y Binarización")
        for boton in (self.btn_nav_norm, self.btn_nav_comp):
            boton.setObjectName("NavButton")
            nav_row.addWidget(boton)
        nav_row.addStretch(1)
        root_layout.addLayout(nav_row)

        self.btn_nav_norm.clicked.connect(lambda: self._mostrar_pagina(0))
        self.btn_nav_comp.clicked.connect(lambda: self._mostrar_pagina(1))

        self.stacked = QStackedWidget()
        self.stacked.addWidget(self._wrap_scroll(self._build_normalizacion_page()))
        self.stacked.addWidget(self._wrap_scroll(self._build_compresion_page()))
        root_layout.addWidget(self.stacked, 1)

        self.stacked.currentChanged.connect(self._actualizar_nav)
        self._actualizar_nav(0)

        self.setCentralWidget(root)

    def _wrap_scroll(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)
        return scroll

    def _crear_card(self):
        frame = QFrame()
        frame.setObjectName("Card")
        return frame

    def _crear_preview(self, alto=150):
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(alto)
        label.setMaximumHeight(alto)
        label.setStyleSheet("background: #FBF7F0; border: 1px solid #E3D9CD; border-radius: 10px; padding: 6px;")
        return label

    def _crear_boton(self, texto, object_name=None):
        boton = QPushButton(texto)
        if object_name: boton.setObjectName(object_name)
        return boton

    def _crear_slider(self, mn, mx, valor, color, callback=None):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(valor)
        slider.setStyleSheet(estilo_slider(color))
        if callback is not None: slider.valueChanged.connect(callback)
        return slider

    def _build_normalizacion_page(self):
        page = QWidget()
        page.setObjectName("PageBg")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 8, 6, 16)
        layout.setSpacing(16)

        encabezado = QLabel("Normalización")
        encabezado.setAlignment(Qt.AlignCenter)
        encabezado.setObjectName("SectionTitle")
        encabezado.setStyleSheet("font-size: 18px; font-weight: 900; color: #4A443F;")
        layout.addWidget(encabezado)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        layout.addLayout(grid)

        resumen_card = self._crear_card()
        resumen_layout = QVBoxLayout(resumen_card)
        resumen_layout.setContentsMargins(16, 16, 16, 16)
        resumen_layout.setSpacing(12)

        resumen_layout.addWidget(QLabel("Imagen Original", alignment=Qt.AlignCenter, objectName="SectionTitle"))
        self.lbl_original_grande = self._crear_preview(150)
        resumen_layout.addWidget(self.lbl_original_grande)

        resumen_layout.addWidget(QLabel("Resultado Normalizado", alignment=Qt.AlignCenter, objectName="AccentGreen"))
        self.lbl_resultado_grande = self._crear_preview(150)
        resumen_layout.addWidget(self.lbl_resultado_grande)

        self.lbl_path = QLabel("", objectName="Muted", wordWrap=True, alignment=Qt.AlignCenter)
        resumen_layout.addWidget(self.lbl_path)

        self.btn_cargar = self._crear_boton("Cargar Imagen", "PrimaryButton")
        self.btn_cargar.clicked.connect(self._seleccionar_imagen)
        resumen_layout.addWidget(self.btn_cargar)
        resumen_layout.addStretch(1)

        grid.addWidget(resumen_card, 0, 0)
        grid.addWidget(self._crear_card_canal("r"), 0, 1)
        grid.addWidget(self._crear_card_canal("g"), 0, 2)
        grid.addWidget(self._crear_card_canal("b"), 0, 3)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1); grid.setColumnStretch(2, 1); grid.setColumnStretch(3, 1)

        acciones = QHBoxLayout()
        acciones.addStretch(1)
        self.btn_reset_todo = self._crear_boton("Reset Todo")
        self.btn_reset_todo.clicked.connect(self._resetear_todo)
        acciones.addWidget(self.btn_reset_todo)
        acciones.addStretch(1)
        layout.addLayout(acciones)

        return page

    def _crear_card_canal(self, key):
        spec = modelo.CHANNEL_SPECS[key]
        card = self._crear_card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14); layout.setSpacing(10)

        titulo = QLabel(spec["title"], alignment=Qt.AlignCenter)
        titulo.setStyleSheet(f"font-size: 15px; font-weight: 900; color: {spec['color']};")
        layout.addWidget(titulo)

        layout.addWidget(QLabel("ORIGINAL", alignment=Qt.AlignCenter, objectName="MiniTitle"))
        lbl_original = self._crear_preview(110)
        layout.addWidget(lbl_original)

        hist_canvas = HistCanvas()
        layout.addWidget(hist_canvas)

        lbl_range = QLabel("", alignment=Qt.AlignCenter, objectName="Muted")
        lbl_range.setStyleSheet("font-weight: 700; color: #7A6F64;")
        layout.addWidget(lbl_range)

        layout.addWidget(QLabel("Min", styleSheet="font-size: 12px; font-weight: 800; color: #6D6358;"))
        slider_min = self._crear_slider(0, 255, 0, spec["color"])
        layout.addWidget(slider_min)

        layout.addWidget(QLabel("Max", styleSheet="font-size: 12px; font-weight: 800; color: #6D6358;"))
        slider_max = self._crear_slider(0, 255, 255, spec["color"])
        layout.addWidget(slider_max)

        layout.addWidget(QLabel("NORMALIZADO", alignment=Qt.AlignCenter, styleSheet=f"font-size: 11px; font-weight: 900; color: {spec['color']}; letter-spacing: 0.5px;"))
        lbl_norm = self._crear_preview(110)
        layout.addWidget(lbl_norm)

        boton = self._crear_boton(f"Limpiar {spec['name']}")
        boton.clicked.connect(lambda _, canal=key: self._resetear_canal(canal))
        layout.addWidget(boton)
        layout.addStretch(1)

        self.channel_widgets[key] = {"original": lbl_original, "normalized": lbl_norm, "hist": hist_canvas, "min": slider_min, "max": slider_max, "range": lbl_range}
        slider_min.valueChanged.connect(lambda _, canal=key: self._on_channel_slider_changed(canal))
        slider_max.valueChanged.connect(lambda _, canal=key: self._on_channel_slider_changed(canal))

        return card

    def _build_compresion_page(self):
        page = QWidget()
        page.setObjectName("PageBg")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 8, 6, 16); layout.setSpacing(16)

        encabezado = QLabel("Compresión y Binarización", alignment=Qt.AlignCenter, objectName="SectionTitle")
        encabezado.setStyleSheet("font-size: 18px; font-weight: 900; color: #4A443F;")
        layout.addWidget(encabezado)

        previews = QGridLayout()
        previews.setHorizontalSpacing(12); previews.setVerticalSpacing(12)
        layout.addLayout(previews)

        self.lbl_comp_resumen = self._crear_panel_resultado("Original Normalizada", "Resolución", alto=190)
        self.lbl_comp_media = self._crear_panel_resultado("Imagen Comprimida (grises)", "Compresión", alto=190)
        self.lbl_comp_binaria = self._crear_panel_resultado("Imagen Binaria (Threshold)", "Binarización", alto=190)

        previews.addWidget(self.lbl_comp_resumen["card"], 0, 0)
        previews.addWidget(self.lbl_comp_media["card"], 0, 1)
        previews.addWidget(self.lbl_comp_binaria["card"], 0, 2)
        previews.setColumnStretch(0, 1); previews.setColumnStretch(1, 1); previews.setColumnStretch(2, 1)

        controles = self._crear_card()
        controles_layout = QVBoxLayout(controles)
        controles_layout.setContentsMargins(16, 16, 16, 16); controles_layout.setSpacing(16)

        controles_layout.addWidget(QLabel("Controles de Procesamiento", objectName="SectionTitle"))
        controles_layout.addWidget(QLabel("Tamaño de bloque (promedio N x N, reduce resolución):", styleSheet="font-size: 13px; font-weight: 800; color: #4A443F;"))

        radios_layout = QHBoxLayout(); radios_layout.setSpacing(24)
        self.block_group = QButtonGroup(self)
        for valor in modelo.BLOCK_OPTIONS:
            radio = QRadioButton(f"{valor}x{valor}")
            radio.toggled.connect(self._on_compresion_params_changed)
            self.block_group.addButton(radio, valor)
            self.block_buttons[valor] = radio
            radios_layout.addWidget(radio)
        radios_layout.addStretch(1)
        controles_layout.addLayout(radios_layout)
        self.block_buttons[2].setChecked(True)

        th_row = QHBoxLayout()
        th_row.addWidget(QLabel("Umbral de binarización (threshold):", styleSheet="font-size: 13px; font-weight: 800; color: #4A443F;"))
        self.lbl_media = QLabel("", objectName="Muted")
        th_row.addWidget(self.lbl_media); th_row.addStretch(1)
        controles_layout.addLayout(th_row)

        slider_row = QHBoxLayout(); slider_row.setSpacing(18)
        self.t_slider = self._crear_slider(0, 255, 128, "#8B7B67", self._on_compresion_params_changed)
        slider_row.addWidget(self.t_slider, 1)
        self.lbl_t = QLabel("Umbral: 128", styleSheet="font-size: 13px; font-weight: 800; color: #4A443F;")
        slider_row.addWidget(self.lbl_t)
        controles_layout.addLayout(slider_row)

        ayuda = QLabel("Los píxeles más claros o iguales al umbral se vuelven blancos; los más oscuros se vuelven negros.", objectName="Muted")
        controles_layout.addWidget(ayuda)
        layout.addWidget(controles)

        acciones = QHBoxLayout()
        acciones.addStretch(1)
        self.btn_guardar = self._crear_boton("Guardar Imagen Binaria", "SuccessButton")
        self.btn_guardar.clicked.connect(self._guardar_binaria)
        acciones.addWidget(self.btn_guardar)
        layout.addLayout(acciones)

        return page

    def _crear_panel_resultado(self, titulo, subtitulo, alto=230):
        card = self._crear_card(); layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
        layout.addWidget(QLabel(titulo, alignment=Qt.AlignCenter, objectName="SectionTitle", styleSheet="font-size: 13px;"))
        imagen = self._crear_preview(alto); layout.addWidget(imagen)
        meta = QLabel(subtitulo, alignment=Qt.AlignCenter, objectName="Muted")
        layout.addWidget(meta)
        return {"card": card, "image": imagen, "meta": meta}

    def _mostrar_pagina(self, index): self.stacked.setCurrentIndex(index)

    def _actualizar_nav(self, index):
        self.btn_nav_norm.setProperty("active", index == 0)
        self.btn_nav_comp.setProperty("active", index == 1)
        for boton in (self.btn_nav_norm, self.btn_nav_comp):
            boton.style().unpolish(boton); boton.style().polish(boton); boton.update()
        if index == 1 and self.ui_ready: self.actualizar_vistas_compresion()

    def _set_label_image(self, label, img_np):
        size = label.size()
        if size.width() <= 10 or size.height() <= 10:
            h = label.minimumHeight(); h = h if h > 0 else 150
            size = QSize(int(h * 1.5), h)
        label.setPixmap(np_to_pixmap(img_np).scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_channel_slider_changed(self, key):
        if not self.ui_ready: return
        s_min, s_max = self.channel_widgets[key]["min"], self.channel_widgets[key]["max"]
        if s_min.value() > s_max.value():
            if self.sender() == s_min: s_max.setValue(s_min.value())
            else: s_min.setValue(s_max.value())
            return
        self.estado.actualizar_rango(key, s_min.value(), s_max.value())
        self.estado.actualizar_normalizacion()
        self.actualizar_vistas_normalizacion()

    def _on_compresion_params_changed(self):
        if not self.ui_ready: return
        bloque = self.block_group.checkedId()
        self.estado.actualizar_compresion(bloque=bloque if bloque > 0 else 2, umbral=self.t_slider.value())
        self.actualizar_vistas_compresion()

    def _resetear_canal(self, key):
        self.channel_widgets[key]["min"].setValue(0)
        self.channel_widgets[key]["max"].setValue(255)
        # Esto disparará _on_channel_slider_changed

    def _resetear_todo(self):
        for key in modelo.CHANNEL_SPECS:
            self.channel_widgets[key]["min"].setValue(0)
            self.channel_widgets[key]["max"].setValue(255)
        self.block_buttons[2].setChecked(True)
        self.t_slider.setValue(128)

    def _seleccionar_imagen(self):
        archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", str(Path.cwd()), "Imágenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if archivo:
            self.estado.establecer_imagen(archivo)
            self.actualizar_vistas_normalizacion()

    def _guardar_binaria(self):
        archivo, _ = QFileDialog.getSaveFileName(self, "Guardar imagen binaria", str(Path.cwd() / "binaria.png"), "PNG (*.png);;JPG (*.jpg *.jpeg);;BMP (*.bmp)")
        if archivo:
            salida = cv2.cvtColor(self.estado.bin_actual, cv2.COLOR_GRAY2BGR)
            ext = Path(archivo).suffix.lower() or ".png"
            exito, buffer = cv2.imencode(ext, salida)
            if exito: buffer.tofile(str(archivo))

    def actualizar_vistas_normalizacion(self):
        for key, spec in modelo.CHANNEL_SPECS.items():
            mn, mx = self.estado.rangos[key]
            widgets = self.channel_widgets[key]
            widgets["range"].setText(f"Min {mn}    Max {mx}")
            self._set_label_image(widgets["original"], self.estado.obtener_canal_original(key))
            self._set_label_image(widgets["normalized"], self.estado.obtener_canal_normalizado(key))
            if self.estado.last_ranges[key] != (mn, mx):
                widgets["hist"].draw_histograms(self.estado.obtener_canal_original(key), self.estado.obtener_canal_normalizado(key), spec["color"])
                self.estado.last_ranges[key] = (mn, mx)
        self._set_label_image(self.lbl_original_grande, self.estado.img)
        self._set_label_image(self.lbl_resultado_grande, self.estado.rgb_norm)
        if self.image_path_label_needed(): self.update_path_label()
        self.actualizar_vistas_compresion()

    def image_path_label_needed(self): return hasattr(self, 'lbl_path')
    def update_path_label(self):
        p = self.estado.image_path
        self.lbl_path.setText(f"Imagen actual: {Path(p).resolve()}" if p else "Usando lienzo vacío.")

    def actualizar_vistas_compresion(self):
        if not self.ui_ready: return
        self._set_label_image(self.lbl_comp_resumen["image"], self.estado.rgb_norm)
        self._set_label_image(self.lbl_comp_media["image"], self.estado.comp_actual)
        self._set_label_image(self.lbl_comp_binaria["image"], self.estado.bin_actual)
        
        ancho, alto = self.estado.img.shape[1], self.estado.img.shape[0]
        bloque = self.estado.bloque
        rw, rh = max(1, ancho // bloque), max(1, alto // bloque)
        reduccion = 100.0 * (1.0 - (rw * rh) / max(1, ancho * alto))
        razon = (ancho * alto) / max(1, rw * rh)
        media = float(np.mean(self.estado.comp_actual))

        self.lbl_comp_resumen["meta"].setText(f"Entrada: {formato_resolucion(self.estado.rgb_norm)} (Normalizada)")
        self.lbl_comp_media["meta"].setText(f"Bloque {bloque}x{bloque} | {ancho}x{alto} px -> {rw}x{rh} px | Reducción: {reduccion:.1f}% | Relación: {razon:.1f}:1")
        self.lbl_comp_binaria["meta"].setText(f"Umbral: {self.estado.umbral} | Media: {media:.1f}")
        self.lbl_t.setText(f"Umbral: {self.estado.umbral}")
        self.lbl_media.setText(f"Media: {media:.1f}")
