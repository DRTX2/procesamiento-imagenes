import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import matplotlib
matplotlib.use('TkAgg')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.color': 'black',
    'font.size': 10
})

# ── CARGAR IMAGEN ───────────────────────────────────
img = cv2.imread('image.png')
if img is None:
    img = np.zeros((300, 400, 3), dtype=np.uint8)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
H, W = img.shape[:2]
gris = (0.299*R + 0.587*G + 0.114*B).astype(np.uint8)

# ── FUNCIONES ───────────────────────────────────────
def interpolar(canal, P):
    f = canal.astype(np.float32)
    S = (P - 128.0) / 2.0
    g = np.where(
        f <= 128,
        S + ((P - S) / 128.0) * f,
        P + ((255.0 - P) / 127.0) * (f - 128)
    )
    return np.clip(g, 0, 255).astype(np.uint8)

def reducir_resolucion(imagen_gris, N):
    if N <= 1:
        return imagen_gris.copy()
    H, W = imagen_gris.shape
    Hr, Wr = (H//N)*N, (W//N)*N
    rec = imagen_gris[:Hr,:Wr].astype(np.float32)
    prom = rec.reshape(Hr//N,N,Wr//N,N).mean(axis=(1,3))
    amp = np.repeat(np.repeat(prom,N,axis=0),N,axis=1)
    res = imagen_gris.copy().astype(np.float32)
    res[:Hr,:Wr] = amp
    return np.clip(res,0,255).astype(np.uint8)

def plot_hist(ax, canal, color, titulo):
    ax.clear()
    ax.hist(canal.flatten(), bins=256, range=(0, 255), color=color, alpha=0.85, linewidth=0)
    ax.set_xlim(0, 255)
    ax.set_title(titulo, fontsize=10, fontweight='bold', pad=4)
    ax.set_ylabel('Nº de píxeles', fontsize=8, color='#555555')
    ax.yaxis.set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(labelsize=7, colors='#555555')
    ax.yaxis.get_offset_text().set_fontsize(7)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_facecolor('#FAFAFA')

# ── FIGURA PRINCIPAL ────────────────────────────────
fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor('#F5F5F5')

# 7 filas: titulo1 | imgs_s1 | histogramas | sliders | titulo2 | imgs_s2 | slider_n
gs = gridspec.GridSpec(
    7, 4,
    height_ratios=[0.18, 2.8, 1.9, 0.22, 0.18, 2.8, 0.22],
    hspace=0.45,
    wspace=0.28,
    left=0.03, right=0.98,
    top=0.97, bottom=0.03
)

# ── TÍTULO SECCIÓN 1 ────────────────────────────────
ax_t1 = fig.add_subplot(gs[0, :])
ax_t1.axis('off')
fig.add_artist(plt.matplotlib.patches.FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0,
    boxstyle="square,pad=0",
    transform=ax_t1.transAxes,
    facecolor='#2C3E50', zorder=0, clip_on=False
))
ax_t1.text(0.012, 0.5, "  Sección 1: Interpolación RGB",
           transform=ax_t1.transAxes,
           fontsize=13, fontweight='bold', color='white',
           va='center', zorder=1)

# ── FILA 1: IMÁGENES ────────────────────────────────
ax_orig = fig.add_subplot(gs[1, 0])
ax_r    = fig.add_subplot(gs[1, 1])
ax_g    = fig.add_subplot(gs[1, 2])
ax_b    = fig.add_subplot(gs[1, 3])

for ax, titulo in zip([ax_orig, ax_r, ax_g, ax_b],
                      ["Original", "Canal R", "Canal G", "Canal B"]):
    ax.axis('off')
    ax.set_title(titulo, fontsize=11, fontweight='bold', pad=6)

ax_orig.imshow(img)
disp_r = ax_r.imshow(R, cmap='gray')
disp_g = ax_g.imshow(G, cmap='gray')
disp_b = ax_b.imshow(B, cmap='gray')

# ── FILA 2: RESULTADO + HISTOGRAMAS ─────────────────
ax_res = fig.add_subplot(gs[2, 0])
ax_hr  = fig.add_subplot(gs[2, 1])
ax_hg  = fig.add_subplot(gs[2, 2])
ax_hb  = fig.add_subplot(gs[2, 3])

ax_res.imshow(img)
ax_res.set_title("Resultado", fontsize=11, fontweight='bold', pad=6)
ax_res.axis('off')

plot_hist(ax_hr, R, '#E74C3C', 'Histograma R')
plot_hist(ax_hg, G, '#27AE60', 'Histograma G')
plot_hist(ax_hb, B, '#2980B9', 'Histograma B')

# ── FILA 3: SLIDERS RGB ─────────────────────────────
ax_sl_r = fig.add_subplot(gs[3, 1])
ax_sl_g = fig.add_subplot(gs[3, 2])
ax_sl_b = fig.add_subplot(gs[3, 3])

slider_r = Slider(ax_sl_r, 'R', 0, 255, valinit=128, color='#E74C3C')
slider_g = Slider(ax_sl_g, 'G', 0, 255, valinit=128, color='#27AE60')
slider_b = Slider(ax_sl_b, 'B', 0, 255, valinit=128, color='#2980B9')

for sl in [slider_r, slider_g, slider_b]:
    sl.label.set_fontsize(10)
    sl.label.set_fontweight('bold')

# ── TÍTULO SECCIÓN 2 ────────────────────────────────
ax_t2 = fig.add_subplot(gs[4, :])
ax_t2.axis('off')
fig.add_artist(plt.matplotlib.patches.FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0,
    boxstyle="square,pad=0",
    transform=ax_t2.transAxes,
    facecolor='#2C3E50', zorder=0, clip_on=False
))
ax_t2.text(0.012, 0.5, "  Sección 2: Reducción de resolución",
           transform=ax_t2.transAxes,
           fontsize=13, fontweight='bold', color='white',
           va='center', zorder=1)

# ── FILA 5: IMÁGENES + INFO ──────────────────────────
ax_gris = fig.add_subplot(gs[5, 0])
ax_red  = fig.add_subplot(gs[5, 1])
ax_info = fig.add_subplot(gs[5, 2:4])

ax_gris.imshow(gris, cmap='gray')
ax_gris.set_title("Escala de Grises", fontsize=11, fontweight='bold', pad=6)
ax_gris.axis('off')

disp_red = ax_red.imshow(gris, cmap='gray')
ax_red.set_title("Imagen Reducida", fontsize=11, fontweight='bold', pad=6)
ax_red.axis('off')

ax_info.axis('off')
for spine in ax_info.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('#BDC3C7')
    spine.set_linewidth(1.5)
ax_info.set_facecolor('#ECF0F1')
ax_info.patch.set_visible(True)
info_txt = ax_info.text(
    0.5, 0.60, '',
    ha='center', va='center',
    fontsize=28, fontweight='bold',
    color='#2C3E50',
    transform=ax_info.transAxes,
    family='monospace'
)
ax_info.text(
    0.5, 0.22, 'Mueve el slider para ajustar el tamaño del bloque',
    ha='center', va='center',
    fontsize=10, color='#95A5A6',
    transform=ax_info.transAxes
)

# ── FILA 6: SLIDER NxN ──────────────────────────────
ax_sl_n = fig.add_subplot(gs[6, 0:3])
slider_n = Slider(ax_sl_n, 'Bloque NxN', 1, 64, valinit=1, valstep=1, color='#2C3E50')
slider_n.label.set_fontsize(10)
slider_n.label.set_fontweight('bold')

# ── UPDATES ─────────────────────────────────────────
def update_rgb(val):
    nR = interpolar(R, slider_r.val)
    nG = interpolar(G, slider_g.val)
    nB = interpolar(B, slider_b.val)

    disp_r.set_data(nR)
    disp_g.set_data(nG)
    disp_b.set_data(nB)

    ax_res.images[0].set_data(np.stack([nR,nG,nB],axis=2))

    plot_hist(ax_hr, nR, '#E74C3C', 'Histograma R')
    plot_hist(ax_hg, nG, '#27AE60', 'Histograma G')
    plot_hist(ax_hb, nB, '#2980B9', 'Histograma B')

    fig.canvas.draw_idle()

def update_muestreo(val):
    N = int(slider_n.val)
    red = reducir_resolucion(gris, N)
    disp_red.set_data(red)

    fig.canvas.draw_idle()

slider_r.on_changed(update_rgb)
slider_g.on_changed(update_rgb)
slider_b.on_changed(update_rgb)
slider_n.on_changed(update_muestreo)

plt.show()