"""
Vision por computadora

Nelson García Bravatti
Joaquín Puente
Diego Linares
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_img(img, title="Imagen", cmap=None):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    # TODO: Matplotlib espera RGB, OpenCV carga BGR.
    # Verifica si la imagen tiene 3 canales y conviértela para visualización correcta.
    if len(img.shape) == 3 and cmap is None:
        img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_show = img
    
    plt.imshow(img_show, cmap=cmap)
    plt.axis('off')
    plt.show()

def manual_contrast_brightness(image, alpha, beta):
    """
    Aplica g(x) = alpha * f(x) + beta de forma segura.
    Args:
        image: numpy array uint8
        alpha: float (contraste)
        beta: float (brillo)
    Returns:
        numpy array uint8
    """
    # 1. Convertir a float32 y normalizar
    img_float = image.astype(np.float32) / 255.0

    # 2. Aplicar contraste y brillo (vectorizado)
    processed = alpha * img_float + (beta / 255.0)

    # 3. Clipping al rango [0, 1]
    processed = np.clip(processed, 0.0, 1.0)

    # 4. Des-normalizar y convertir a uint8
    processed_img = (processed * 255).astype(np.uint8)
    
    return processed_img

def manual_gamma_correction(image, gamma):
    """
    Aplica V_out = V_in ^ gamma
    """
    # 1. Convertir a float32 y normalizar
    img_float = image.astype(np.float32) / 255.0

    # 2. Corrección gamma (vectorizada)
    gamma_corrected = np.power(img_float, gamma)

    # 3. Clipping por seguridad
    gamma_corrected = np.clip(gamma_corrected, 0.0, 1.0)

    # 4. Volver a uint8
    gamma_img = (gamma_corrected * 255).astype(np.uint8)
    return gamma_img

def hsv_segmentation(image):
    """
    Segmentar un objeto de color específico (ej. verde o rojo)
    """
    # 1. Convertir a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # RETO 3: Definir rangos para un color.
    # OJO: En OpenCV Hue es [0, 179].
    # Ejemplo: Si buscas verde, H está alrededor de 60 (en escala 0-179).
    
    # TODO: Definir lower_bound y upper_bound (np.array)
    lower_bound = np.array([0, 0, 0]) 
    upper_bound = np.array([0, 0, 0])
    
    # Crear máscara
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Aplicar máscara a la imagen original (bitwise_and)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    # Cargar imagen (Asegúrate de tener una imagen 'sample.jpg')
    img = cv2.imread('copa.jpg')
    
    if img is None:
        print("Error: No se encontró la imagen.")
    else:
        # 1. Prueba de Contraste
        contrast_img = manual_contrast_brightness(img, 1.5, 20)
        show_img(contrast_img, "Contraste Alto (Manual)")
        
        # 2. Prueba de Gamma
        gamma_img = manual_gamma_correction(img, 0.5) # Aclarar sombras
        show_img(gamma_img, "Corrección Gamma 0.5")
        
        # 3. Segmentación
        seg_img = hsv_segmentation(img)
        show_img(seg_img, "Segmentación HSV")