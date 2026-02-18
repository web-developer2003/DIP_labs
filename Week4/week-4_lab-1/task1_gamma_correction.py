import cv2
import numpy as np
import matplotlib.pyplot as plt


def gamma_correction(c, gamma, I_in):
    
    # Step 1 – normalize to [0, 1]
    I_norm = I_in.astype('float32') / 255

    # Step 2 – apply power law:  T(r) = c * r^gamma
    I_transformed = c * (I_norm ** gamma)

    # Step 3 – clip to [0, 1] in case c > 1 pushes values above 1
    I_transformed = np.clip(I_transformed, 0, 1)

    # Step 4 – convert back to uint8 for display / saving
    I_out = (I_transformed * 255).astype('uint8')

    return I_out


img1 = cv2.imread("dark.jpg")   # dark image  → needs gamma < 1  (e.g. 0.4) to brighten
img2 = cv2.imread("light.jpg")   # bright image → needs gamma > 1  (e.g. 2.5) to darken
img3 = cv2.imread("lowjpg.jpg")   # low-contrast → try gamma ~ 0.5
img4 = cv2.imread("overexposed.jpg")   # overexposed  → try gamma ~ 2.0
img5 = cv2.imread("normaljpg.jpg")   # normal image → try gamma = 1.0 (no change)

images = [img1, img2, img3, img4, img5]
titles = ["Image 1", "Image 2", "Image 3", "Image 4", "Image 5"]

# Gamma values chosen per image – adjust after visual inspection
gammas = [0.4, 2.5, 0.5, 2.0, 1.0]
c = 1  # scaling factor


for i, (img, gamma, title) in enumerate(zip(images, gammas, titles)):
    if img is None:
        print(f"[WARNING] {title}: image not loaded, skipping.")
        continue

    output = gamma_correction(c, gamma, img)

    # Convert BGR → RGB for matplotlib display
    img_rgb    = cv2.cvtColor(img,    cv2.COLOR_BGR2RGB)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_rgb);    axes[0].set_title(f"{title} – Original");       axes[0].axis('off')
    axes[1].imshow(output_rgb); axes[1].set_title(f"{title} – γ={gamma}, c={c}"); axes[1].axis('off')
    plt.suptitle(
        f"Gamma choice rationale:\n"
        f"  γ<1 brightens (used for dark images) | γ>1 darkens (used for bright/overexposed images)",
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(f"task1_output_image{i+1}.png", dpi=150)
    plt.show()

    print(f"{title}: γ={gamma} chosen because ", end="")
    if gamma < 1:
        print("the image is dark/underexposed – γ<1 expands low intensities.")
    elif gamma > 1:
        print("the image is bright/overexposed – γ>1 compresses high intensities.")
    else:
        print("γ=1 leaves the image unchanged (baseline test).")


if img1 is not None:
    I_norm = img1.astype('float32') / 255

    # --- PROBLEM: saving float32 directly ---
    cv2.imwrite("task1_normalized_BROKEN.png", I_norm)
    # Result: image appears almost completely black because pixel values
    # are in [0, 1] but imwrite interprets float images as [0, 255].

    # --- FIX: convert back to uint8 before saving ---
    I_norm_uint8 = (I_norm * 255).astype('uint8')
    cv2.imwrite("task1_normalized_FIXED.png", I_norm_uint8)
    print("\nTask 1-3: Saved fixed normalized image as 'task1_normalized_FIXED.png'")
    print("Problem : cv2.imwrite treats float32 values as 0–255, so [0,1] → near-black.")
    print("Fix     : multiply by 255 and cast to uint8 before calling imwrite.")
