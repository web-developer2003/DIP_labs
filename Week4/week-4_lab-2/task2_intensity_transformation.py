import cv2
import numpy as np
import matplotlib.pyplot as plt


# ─── Load image (grayscale for intensity operations) ─────────────────────────
img_bgr  = cv2.imread("team2.jpg")
img_gray = cv2.imread("team2.jpg", cv2.IMREAD_GRAYSCALE)

if img_bgr is None:
    print("[ERROR] team2.jpg not found.")
    exit()


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1 – Image Negative
# ═══════════════════════════════════════════════════════════════════════════════
def image_negative(img):
    """s = L - 1 - r   (works for both grayscale uint8 and color uint8)"""
    return 255 - img


neg = image_negative(img_gray)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img_gray, cmap='gray'); axes[0].set_title("Original (Gray)"); axes[0].axis('off')
axes[1].imshow(neg,      cmap='gray'); axes[1].set_title("Negative (s = 255 – r)"); axes[1].axis('off')
plt.suptitle("Task 1 – Image Negative")
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2 – Power-law (Gamma) Curves
# ═══════════════════════════════════════════════════════════════════════════════
gammas = [0.02, 0.2, 0.5, 0.8, 1, 2, 3, 3.5, 4]
c = 1

# ── (a) Non-normalized: x in [0, 255], s = c * r^gamma ────────────────────
x_raw = np.arange(0, 256, dtype=np.float32)

fig, ax = plt.subplots(figsize=(7, 6))
for g in gammas:
    y = c * (x_raw ** g)
    ax.plot(x_raw, y, label=f"γ={g}")
ax.set_xlabel("Input intensity r  (0 – 255)")
ax.set_ylabel("Output intensity s = c·rᵞ")
ax.set_title("(a) Non-normalized gamma curves\n"
             "Curves do NOT meet at y=255 (except γ=1) because 255^γ ≠ 255 when γ≠1")
ax.legend(fontsize=7, loc='upper left')
ax.set_xlim(0, 255); ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Why curves don't meet at 255:
# s(255) = c * 255^gamma.  Only when gamma=1 does this equal 255.
# For gamma<1: 255^gamma < 255 (output is LESS than 255)
# For gamma>1: 255^gamma > 255 (output exceeds 255 – gets clipped if applied to images)

# ── (b) Normalized: x in [0, 1], s = c * r^gamma ─────────────────────────
x_norm = np.arange(0, 1.001, 0.001, dtype=np.float32)

fig, ax = plt.subplots(figsize=(7, 6))
for g in gammas:
    y = c * (x_norm ** g)
    ax.plot(x_norm, y, label=f"γ={g}")
ax.set_xlabel("Input intensity r  (0 – 1, normalized)")
ax.set_ylabel("Output intensity s = c·rᵞ")
ax.set_title("(b) Normalized gamma curves\n"
             "All curves meet at (1, 1) because 1^γ = 1 for any γ")
ax.legend(fontsize=7, loc='upper left')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Why normalized curves all meet at (1,1):
# s(1) = c * 1^gamma = 1 * 1 = 1  — true for ANY value of gamma.
# Normalizing ensures the domain and range are both [0,1],
# so the math forces every curve through (0,0) and (1,1).


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3 – Gamma Correction on the image
# ═══════════════════════════════════════════════════════════════════════════════
def gamma_correction(img, gamma, c=1):
    """Apply s = c * (r/255)^gamma, then scale back to [0,255]."""
    norm       = img.astype(np.float32) / 255.0
    corrected  = np.clip(c * (norm ** gamma), 0, 1)
    return (corrected * 255).astype(np.uint8)


demo_gammas = [0.4, 1.0, 2.5]          # darken < 1=neutral, lighten > 1
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, len(demo_gammas) + 1, figsize=(14, 4))
axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis('off')
for i, g in enumerate(demo_gammas):
    out = gamma_correction(img_bgr, g)
    axes[i + 1].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    axes[i + 1].set_title(f"γ = {g}")
    axes[i + 1].axis('off')
plt.suptitle("Task 3 – Gamma Correction (c=1)")
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Task 4 – Highlight intensities in range [A, B] → set to L-1 = 255
# ═══════════════════════════════════════════════════════════════════════════════
def highlight_range(img_gray, A, B):
    """
    Returns a copy of img_gray where every pixel whose intensity falls
    in [A, B] is set to L-1 = 255.  All other pixels are unchanged.
    """
    out = img_gray.copy()
    mask = (img_gray >= A) & (img_gray <= B)   # equivalent to Matlab's find()
    out[mask] = 255
    return out


A, B = 100, 180          # example range – adjust as needed
highlighted = highlight_range(img_gray, A, B)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img_gray,    cmap='gray'); axes[0].set_title("Original Gray");               axes[0].axis('off')
axes[1].imshow(highlighted, cmap='gray'); axes[1].set_title(f"Intensities [{A},{B}] → 255"); axes[1].axis('off')
plt.suptitle("Task 4 – Range Highlighting ([A,B] → L-1=255)")
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Task 5 – Appropriate transformation to enhance the image
# ═══════════════════════════════════════════════════════════════════════════════
# Inspect the image histogram first
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img_gray, cmap='gray'); axes[0].set_title("Original"); axes[0].axis('off')
axes[1].hist(img_gray.ravel(), bins=256, range=(0, 255), color='gray')
axes[1].set_title("Histogram"); axes[1].set_xlabel("Intensity"); axes[1].set_ylabel("Count")
plt.suptitle("Task 5 – Histogram Inspection (choose transformation based on this)")
plt.tight_layout()
plt.show()

# Auto-select a reasonable gamma based on mean brightness
mean_val = img_gray.mean()
if mean_val < 100:
    chosen_gamma = 0.5          # dark image → brighten
elif mean_val > 170:
    chosen_gamma = 2.0          # bright image → darken
else:
    chosen_gamma = 1.0          # balanced → identity

enhanced = gamma_correction(img_bgr, chosen_gamma)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img_rgb); axes[0].set_title(f"Original (mean={mean_val:.1f})"); axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Enhanced with γ={chosen_gamma}"); axes[1].axis('off')
plt.suptitle("Task 5 – Image Enhancement")
plt.tight_layout()
plt.show()
