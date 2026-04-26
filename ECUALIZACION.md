# Proyecto: Implementación de Ecualización de Histograma

## 1. Objetivo del Sistema

El objetivo es modificar el script de procesamiento de imágenes existente para reemplazar la función de "Interpolación" por una **Ecualización de Histograma**. Esta técnica redistribuye los niveles de intensidad de los píxeles para mejorar el contraste global de una imagen. A diferencia de la expansión de contraste (Min-Max), que es lineal, la ecualización es un proceso no lineal que busca aplanar el histograma de la imagen.

## 2. Conceptos Clave

### Histograma
Es un gráfico que muestra la frecuencia de cada nivel de intensidad (de 0 a 255) en la imagen. Un pico alto significa que muchos píxeles tienen ese tono.

### Histograma Acumulado (CDF - Cumulative Distribution Function)
Es el corazón de la ecualización. Para cada nivel de intensidad `i`, la CDF nos dice cuántos píxeles tienen una intensidad menor o igual a `i`. Es una función que siempre va en aumento.

### Fórmula de Ecualización
La idea es usar la CDF para "mapear" los valores de píxeles originales a nuevos valores. La fórmula simplificada para cada píxel es:

`nuevo_valor = (CDF(valor_original) - CDF_min) / (Total_pixeles - CDF_min) * 255`

Donde:
- `CDF(valor_original)`: El valor del histograma acumulado para el nivel de gris del píxel.
- `CDF_min`: El primer valor distinto de cero en la CDF (el valor acumulado para el tono más oscuro presente en la imagen).
- `Total_pixeles`: El número total de píxeles en la imagen (Ancho x Alto).

## 3. Requisitos y Pasos de Implementación

### 3.1. Modificar la Interfaz Gráfica
La ecualización es generalmente un proceso automático que se aplica a toda la imagen. Por lo tanto, los 3 sliders de RGB ya no tienen sentido para esta función.

*   **Acción:** Elimina los tres sliders de `R`, `G` y `B`.
*   **Reemplazo:** Añade un único botón con el texto "Ecualizar Histograma".

### 3.2. Crear la Función de Ecualización
Necesitarás una nueva función en Python que realice la ecualización. OpenCV ya tiene una función integrada que hace esto perfectamente: `cv2.equalizeHist()`. Usar esta función es mucho más eficiente que implementarla desde cero.

**Firma de la función en OpenCV:**
`dst = cv2.equalizeHist(src)`
- `src`: La imagen de entrada en escala de grises (de un solo canal y 8 bits).
- `dst`: La imagen de salida ecualizada.

### 3.3. Lógica del Programa

El flujo de trabajo al presionar el nuevo botón "Ecualizar Histograma" debería ser:

1.  **Separar Canales:** Tomar la imagen original y dividirla en sus tres canales: R, G, B.
2.  **Ecualizar cada Canal:** Aplicar la función `cv2.equalizeHist()` a cada uno de los canales (R, G, B) de forma independiente.
    ```python
    r_ecualizado = cv2.equalizeHist(canal_R)
    g_ecualizado = cv2.equalizeHist(canal_G)
    b_ecualizado = cv2.equalizeHist(canal_B)
    ```
3.  **Unir Canales:** Volver a combinar los tres canales ecualizados para formar una nueva imagen a color.
    ```python
    imagen_resultado = cv2.merge([r_ecualizado, g_ecualizado, b_ecualizado])
    ```
4.  **Actualizar la Vista:**
    *   Mostrar la `imagen_resultado` en el panel "Resultado".
    *   Mostrar cada canal ecualizado (`r_ecualizado`, `g_ecualizado`, `b_ecualizado`) en sus respectivos paneles.
    *   Actualizar los histogramas para que reflejen la nueva distribución de píxeles de los canales ecualizados. Notarás que los nuevos histogramas están mucho más "esparcidos" y planos.

## 4. Consideraciones Adicionales

*   **Color vs. Escala de Grises:** La ecualización directa en canales RGB puede a veces producir cambios de color extraños. Un método alternativo (y a menudo mejor) es convertir la imagen a un espacio de color como YCrCb o HSV, ecualizar solo el canal de luminancia/brillo (Y o V) y luego volver a convertir a RGB. Por simplicidad, empezar con la ecualización de canales RGB separados es un buen primer paso.
*   **OpenCV es tu amigo:** Re-implementar la CDF y la fórmula de ecualización manualmente es un gran ejercicio académico, pero para una aplicación práctica, `cv2.equalizeHist()` es la herramienta correcta, ya que está optimizada y probada.

---
**Resumen para tu código:** Deberás eliminar los sliders y su función `update_rgb`, añadir un botón, y crear una nueva función que se active con ese botón y que use `cv2.equalizeHist()` en cada canal de color para luego actualizar la interfaz.
