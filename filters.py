import cv2
import numpy as np

def apply_filters(img):
    results = {}

    # Box Filter
    results['box'] = cv2.blur(img, (5, 5))

    # Gaussian Filter
    results['gaussian'] = cv2.GaussianBlur(img, (5, 5), 0)

    # Median Filter
    results['median'] = cv2.medianBlur(img, 5)

    # Sobel
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = cv2.magnitude(sobelx, sobely)

    results['sobel'] = sobel.astype(np.uint8)

    return results

def gradient_direction_demo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    patch = gray[h//2:h//2+100, w//2:w//2+100]

    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)

    return patch, magnitude, direction, gx, gy

def fft_spectrum(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    return fshift, magnitude_spectrum

def low_pass_filter(fshift):
    rows, cols = fshift.shape
    crow, ccol = rows//2, cols//2

    mask = np.zeros_like(fshift)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    fshift_filtered = fshift * mask
    return fshift_filtered

def draw_gradient_arrows(patch, gx, gy, step=10, scale=0.3):
   
    vis = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

    for i in range(0, patch.shape[0], step):
        for j in range(0, patch.shape[1], step):
            dx = gx[i, j]
            dy = gy[i, j]

            end_x = int(j + dx * scale)
            end_y = int(i + dy * scale)

            cv2.arrowedLine(vis, (j, i), (end_x, end_y), (0, 0, 255), 1, tipLength=0.3)

    return vis
