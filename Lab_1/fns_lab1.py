import numpy as np
import cv2


def mi_convolucion(image, kernel, padding_type='reflect'):
    pass

def generar_gaussiano(size, sigma):
    pass

def detectar_border_sobel(image):
    pass


if __name__ = '__main__':
    img = cv2.imread()
    
    if img is None:
        print("Error: No se encontró la imagen.")
    else:
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mi_convolucion(grayscale, kernel)

        generar_gaussiano

        detectar_border_sobel(img)

