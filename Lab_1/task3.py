import cv2
import numpy as np

# Helpers

def to_gray(img_bgr):
    if len(img_bgr.shape) == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr

def normalize_0_255(img_float):
    img = img_float.astype(np.float32)
    img -= img.min()
    maxv = img.max()
    if maxv > 0:
        img = img / maxv
    return (img * 255).astype(np.uint8)

def add_gaussian_noise(img_gray, sigma_noise=15):
    noise = np.random.normal(0, sigma_noise, img_gray.shape).astype(np.float32)
    noisy = img_gray.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# A: Gaussian + Sobel Magnitude
def sobel_magnitude(img_gray):
    # Sobel gradients (OpenCV)
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return normalize_0_255(mag)

def run_experiment_a(img_gray):
    # A1: No smoothing
    mag_no = sobel_magnitude(img_gray)

    # A2: sigma=1, k=5
    blur_s1 = cv2.GaussianBlur(img_gray, (5, 5), sigmaX=1, sigmaY=1)
    mag_s1 = sobel_magnitude(blur_s1)

    # A3: sigma=5, k=31
    blur_s5 = cv2.GaussianBlur(img_gray, (31, 31), sigmaX=5, sigmaY=5)
    mag_s5 = sobel_magnitude(blur_s5)

    return mag_no, mag_s1, mag_s5

# B: Simple threshold vs Canny
def simple_threshold(mag_0_255, T):
    out = np.zeros_like(mag_0_255, dtype=np.uint8)
    out[mag_0_255 >= T] = 255
    return out

def run_experiment_b(img_gray, T=80, canny_low=50, canny_high=150):
    mag = sobel_magnitude(img_gray)
    thr = simple_threshold(mag, T)
    canny = cv2.Canny(img_gray, canny_low, canny_high)
    return mag, thr, canny


# Main 
path = "carretera.jpg" 
img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(f"No pude leer la imagen: {path}")

gray = to_gray(img)

gray = add_gaussian_noise(gray, sigma_noise=15)

# Run A
mag_no, mag_s1, mag_s5 = run_experiment_a(gray)

# Run B
mag, thr, canny = run_experiment_b(gray, T=80, canny_low=50, canny_high=150)

cv2.imwrite("A_ruido.png", mag_no)
cv2.imwrite("A_ruido_sigma1.png", mag_s1)
cv2.imwrite("A_ruido_sigma5.png", mag_s5)

cv2.imwrite("B_ruido.png", mag)
cv2.imwrite("B_umbral.png", thr)
cv2.imwrite("B_canny.png", canny)

