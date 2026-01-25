import numpy as np
import cv2


def mi_convolucion(imagen, kernel, padding_type='reflect'):
    kernel = np.flipud(np.fliplr(kernel))

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    imagen_padded = np.pad(imagen.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode=padding_type)

    h, w = imagen.shape
    output = np.zeros((h, w), dtype=np.float64)

    for i in range(kh):
        for j in range(kw):
            output += kernel[i, j] * imagen_padded[i:i+h, j:j+w]

    return output


def generar_gaussiano(tamano, sigma):
    centro = tamano // 2
    x = np.arange(tamano) - centro
    y = np.arange(tamano) - centro
    xx, yy = np.meshgrid(x, y)

    gaussiano = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussiano /= gaussiano.sum()

    return gaussiano


def detectar_bordes_sobel(imagen):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float64)

    gx = mi_convolucion(imagen, sobel_x)
    gy = mi_convolucion(imagen, sobel_y)

    magnitud = np.sqrt(gx**2 + gy**2)
    magnitud = (magnitud / magnitud.max()) * 255
    magnitud = magnitud.astype(np.uint8)

    direccion = np.arctan2(gy, gx)

    return magnitud, direccion


if __name__ == '__main__':
    img = cv2.imread('test_image.jpg')

    if img is None:
        print("Error: No se encontró la imagen.")
    else:
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel_ejemplo = generar_gaussiano(5, 1.0)
        img_suavizada = mi_convolucion(grayscale, kernel_ejemplo)

        magnitud, direccion = detectar_bordes_sobel(grayscale)

        cv2.imshow('Original', grayscale)
        cv2.imshow('Suavizada', img_suavizada.astype(np.uint8))
        cv2.imshow('Bordes Sobel', magnitud)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

