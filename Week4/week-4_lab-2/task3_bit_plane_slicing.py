import cv2
import numpy as np
import matplotlib.pyplot as plt


def displayBitResponse(img):
    """
    Takes a grayscale image (uint8) and displays all 8 bit-plane images.

    Each pixel value is 8 bits: b7 b6 b5 b4 b3 b2 b1 b0
      - bit 0 (LSB) = bit plane 1 → least significant, mostly noise
      - bit 7 (MSB) = bit plane 8 → most significant, carries structural info

    For each plane k (1..8):
        plane = (img >> (k-1)) & 1        # extract k-th bit
    Multiply by 255 so the binary image is visible (0 or 255).
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for k in range(1, 9):                        # k = 1 (LSB) … 8 (MSB)
        bit_plane = ((img >> (k - 1)) & 1) * 255
        bit_plane = bit_plane.astype(np.uint8)

        axes[k - 1].imshow(bit_plane, cmap='gray')
        axes[k - 1].set_title(f"bit plane{k}")
        axes[k - 1].axis('off')

    plt.suptitle("Bit-Plane Slicing – All 8 Planes\n"
                 "(Plane 1 = LSB, noisy | Plane 8 = MSB, structurally dominant)", y=1.02)
    plt.tight_layout()
    plt.show()


# ─── Load and run ─────────────────────────────────────────────────────────────
img = cv2.imread("team2.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("[ERROR] team2.jpg not found. Place it in the same directory.")
else:
    displayBitResponse(img)
