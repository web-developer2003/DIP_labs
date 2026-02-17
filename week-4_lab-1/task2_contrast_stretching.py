import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(i_in, smax, smin):
    
    img = i_in.astype('float32')

    rmin = float(img.min())
    rmax = float(img.max())

    # Avoid division by zero when image is completely flat
    if rmax == rmin:
        print("[WARNING] rmax == rmin: image has uniform intensity, returning smin-filled image.")
        return np.full_like(img, smin, dtype='float32')

    # Apply formula: s = ((smax - smin) / (rmax - rmin)) * (r - rmin) + smin
    stretched = ((smax - smin) / (rmax - rmin)) * (img - rmin) + smin

    return stretched.astype('float32')


def show_and_save(original, result, title, filename, smax, smin, normalized=False):
    """Helper: display side-by-side and save figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Determine display range for result
    vmin_disp = min(smin, smax)
    vmax_disp = max(smin, smax)

    cmap = 'gray' if len(original.shape) == 2 else None

    if cmap:
        axes[0].imshow(original, cmap='gray', vmin=0, vmax=255 if not normalized else 1)
        axes[1].imshow(result,   cmap='gray', vmin=vmin_disp, vmax=vmax_disp)
    else:
        orig_disp = cv2.cvtColor(original.astype('uint8') if original.max() > 1
                                 else (original * 255).astype('uint8'),
                                 cv2.COLOR_BGR2RGB)
        res_clipped = np.clip(result, vmin_disp, vmax_disp)
        res_disp  = cv2.cvtColor(
            ((res_clipped - vmin_disp) / max(vmax_disp - vmin_disp, 1e-6) * 255).astype('uint8'),
            cv2.COLOR_BGR2RGB
        )
        axes[0].imshow(orig_disp)
        axes[1].imshow(res_disp)

    axes[0].set_title("Original" + (" (normalized)" if normalized else "")); axes[0].axis('off')
    axes[1].set_title(f"{title}\nsmax={smax}, smin={smin}");                  axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")
    
img1 = cv2.imread("dark.jpg")   # dark image  → needs gamma < 1  (e.g. 0.4) to brighten
img2 = cv2.imread("light.jpg")   # bright image → needs gamma > 1  (e.g. 2.5) to darken
img3 = cv2.imread("lowjpg.jpg")   # low-contrast → try gamma ~ 0.5
img4 = cv2.imread("overexposed.jpg")   # overexposed  → try gamma ~ 2.0
img5 = cv2.imread("normaljpg.jpg") 

images = [img1, img2, img3, img4, img5]
names  = ["img1", "img2", "img3", "img4", "img5"]


# ─────────────────────────────────────────────
#  Task 2-1 : Non-normalized, smax=0, smin=255
#  (swapping min/max intentionally inverts image)
# ─────────────────────────────────────────────

print("=" * 60)
print("Task 2-1 : Non-normalized | smax=0, smin=255")
print("=" * 60)

for img, name in zip(images, names):
    if img is None:
        print(f"[WARNING] {name}: not loaded, skipping.")
        continue

    result = contrast_stretching(img, smax=0, smin=255)
    show_and_save(img, result,
                  title=f"Contrast Stretch ({name})",
                  filename=f"task2_1_{name}_smax0_smin255.png",
                  smax=0, smin=255)

print("\nObservation: smax=0, smin=255 INVERTS the image because the minimum input")
print("is mapped to 255 and the maximum input is mapped to 0.")


# ─────────────────────────────────────────────
#  Task 2-2 : Normalized images, smax=0, smin=1
# ─────────────────────────────────────────────

print("=" * 60)
print("Task 2-2 : Normalized [0,1] | smax=0, smin=1")
print("=" * 60)

for img, name in zip(images, names):
    if img is None:
        print(f"[WARNING] {name}: not loaded, skipping.")
        continue

    img_norm = img.astype('float32') / 255        # normalize to [0, 1]
    result   = contrast_stretching(img_norm, smax=0, smin=1)
    show_and_save(img_norm, result,
                  title=f"Normalized Contrast Stretch ({name})",
                  filename=f"task2_2_{name}_normalized_smax0_smin1.png",
                  smax=0, smin=1, normalized=True)


# ─────────────────────────────────────────────
#  Task 2-3 : Does normalization help?
# ─────────────────────────────────────────────

print("=" * 60)
print("Task 2-3 : Effect of normalization")
print("=" * 60)
print("""
Observation:
- Non-normalized (uint8, range 0-255) with smax=0, smin=255:
  The contrast stretching formula still works correctly because rmin/rmax
  are computed from the actual pixel values.  The output range is [0, 255]
  (inverted), which OpenCV/matplotlib can display directly.

- Normalized (float32, range 0-1) with smax=0, smin=1:
  The output range is [0, 1] which also inverts the image.
  Numerically the two are equivalent (just scaled differently).

- Impact: Normalization itself does NOT improve or hurt the stretching
  quality here – both produce the same visual inversion.
  Normalization is more useful when combining operations (e.g. gamma
  followed by stretching) where intermediate values must stay in [0,1].
""")


# ─────────────────────────────────────────────
#  Task 2-4 : Various smax / smin combinations
# ─────────────────────────────────────────────

print("=" * 60)
print("Task 2-4 : Various combinations (non-normalized)")
print("=" * 60)

combos = [
    (0,   50,  "Low range – very dark output, most detail compressed into [0,50]"),
    (100, 160, "Narrow mid-range – low contrast, output squeezed into [100,160]"),
    (0,   255, "Full standard stretch – maximum contrast enhancement"),
    (50,  200, "Partial stretch – moderate contrast boost"),
]

for img, name in zip(images, names):
    if img is None:
        continue
    for smax, smin, observation in combos:
        result = contrast_stretching(img, smax=smax, smin=smin)
        filename = f"task2_4_{name}_smax{smax}_smin{smin}.png"
        show_and_save(img, result,
                      title=f"Stretch ({name})",
                      filename=filename,
                      smax=smax, smin=smin)
        print(f"  smax={smax:3d}, smin={smin:3d} → {observation}")

print("""
Summary of observations for Task 2-4:
  (smax=0,   smin=50)  : Output range [0,50]  → very dark, low visibility.
                          Histogram is compressed toward black; fine detail lost.
  (smax=100, smin=160) : Output range [100,160] → narrow mid-gray band,
                          very low contrast – image looks washed-out / flat.
  (smax=0,   smin=255) : Full inversion & stretch – maximum contrast, inverted.
  (smax=50,  smin=200) : Moderate stretch – reasonable contrast improvement
                          without pushing extremes.
""")
