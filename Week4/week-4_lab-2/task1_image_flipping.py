import cv2
import numpy as np
import matplotlib.pyplot as plt


# ─── Task 1: Color → Grayscale ───────────────────────────────────────────────
def color_to_grayscale(img):
    """Takes a color BGR image and returns its grayscale version."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─── Task 2: Mirror Effect ────────────────────────────────────────────────────
def mirror_effect(img):
    """Takes an image and returns [original | horizontally-flipped] side by side."""
    flipped = cv2.flip(img, 1)          # flip along vertical axis (horizontal mirror)
    return np.concatenate([img, flipped], axis=1)   # equivalent to Matlab [A B]


# ─── Load image ──────────────────────────────────────────────────────────────
img = cv2.imread("team2.jpg")

if img is None:
    print("[ERROR] team2.jpg not found. Place it in the same directory.")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Task 1 display ──────────────────────────────────────────────────────
    gray = color_to_grayscale(img)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_rgb);   axes[0].set_title("Input: Color Image");     axes[0].axis('off')
    axes[1].imshow(gray, cmap='gray'); axes[1].set_title("Output: Grayscale"); axes[1].axis('off')
    plt.suptitle("Task 1 – Color Image → Grayscale")
    plt.tight_layout()
    plt.show()

    # ── Task 2 display ──────────────────────────────────────────────────────
    mirrored     = mirror_effect(img)
    mirrored_rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(img_rgb);      axes[0].set_title("Input Image");                  axes[0].axis('off')
    axes[1].imshow(mirrored_rgb); axes[1].set_title("Mirror Effect [img | flipped]"); axes[1].axis('off')
    plt.suptitle("Task 2 – Mirror Effect")
    plt.tight_layout()
    plt.show()
