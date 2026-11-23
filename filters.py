import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import heapq
from collections import Counter
from decimal import Decimal, getcontext

# Load standard 512x512 grayscale DIP test image
img = data.camera()

plt.figure(figsize=(4, 4))
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()


# 1. Mean Filter (5x5 averaging)
mean_img = cv2.blur(img, (5, 5))

# 2. Weighted Average Filter (3x3 standard kernel)
weighted_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], dtype=np.float32) / 16.0
weighted_img = cv2.filter2D(img, -1, weighted_kernel)

# 3. Gaussian Filter
gaussian_img = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)

# 4. Median Filter
median_img = cv2.medianBlur(img, 5)

# 5. Laplacian Filter (Edge Detection)
laplacian_raw = cv2.Laplacian(img, cv2.CV_64F)
laplacian_img = cv2.convertScaleAbs(laplacian_raw)

# 6. Sobel Filter (Combined X and Y gradient magnitude)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_img = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))

# 7. High-Boost Filter: Original + k * (Original - Blurred), where k > 1
k = 2.0
blurred = cv2.GaussianBlur(img, (5, 5), 1.5).astype(np.float32)
mask = img.astype(np.float32) - blurred
high_boost_img = np.clip(img.astype(np.float32) + k * mask, 0, 255).astype(np.uint8)

# --- Plot Spatial Filters ---
spatial_results = [
    ("Original", img),
    ("Mean (5x5)", mean_img),
    ("Weighted Avg (3x3)", weighted_img),
    ("Gaussian (5x5)", gaussian_img),
    ("Median (5x5)", median_img),
    ("Laplacian", laplacian_img),
    ("Sobel Magnitude", sobel_img),
    (f"High-Boost (k={k})", high_boost_img),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, (title, im) in zip(axes.ravel(), spatial_results):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()





def apply_freq_filter(image, H):
    # Forward 2D FFT and shift zero-frequency component to center
    F = np.fft.fftshift(np.fft.fft2(image))
    # Multiply in frequency domain
    G = F * H
    # Inverse shift and Inverse 2D FFT
    filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(G)))
    return filtered

# Distance matrix D(u, v) from center of frequency rectangle
M, N = img.shape
u = np.arange(M) - M // 2
v = np.arange(N) - N // 2
V, U = np.meshgrid(v, u)
D = np.sqrt(U**2 + V**2)

D0 = 40  # Cutoff frequency
n = 2    # Butterworth order

# 1. Ideal LPF & HPF
H_ideal_lpf = (D <= D0).astype(np.float64)
H_ideal_hpf = 1.0 - H_ideal_lpf

# 2. Butterworth LPF & HPF
H_butter_lpf = 1.0 / (1.0 + (D / D0)**(2 * n))
H_butter_hpf = 1.0 - H_butter_lpf

# 3. Gaussian LPF & HPF
H_gauss_lpf = np.exp(-(D**2) / (2 * (D0**2)))
H_gauss_hpf = 1.0 - H_gauss_lpf

# Apply all 6 frequency filters
freq_results = [
    (f"Ideal LPF (D0={D0})", apply_freq_filter(img, H_ideal_lpf)),
    (f"Butterworth LPF (D0={D0}, n={n})", apply_freq_filter(img, H_butter_lpf)),
    (f"Gaussian LPF (D0={D0})", apply_freq_filter(img, H_gauss_lpf)),
    (f"Ideal HPF (D0={D0})", apply_freq_filter(img, H_ideal_hpf)),
    (f"Butterworth HPF (D0={D0}, n={n})", apply_freq_filter(img, H_butter_hpf)),
    (f"Gaussian HPF (D0={D0})", apply_freq_filter(img, H_gauss_hpf)),
]

# --- Plot Frequency Filters ---
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, (title, im) in zip(axes.ravel(), freq_results):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()

