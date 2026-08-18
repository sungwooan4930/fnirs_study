"""
Generate a realistic front-facing (anterior/frontal) brain image for the fNIRS brain map widget.
Shows the prefrontal cortex as seen from the front (like looking at a patient's forehead).
Saves brain_top.png (600x600, RGBA) in this directory.

Run once:
    python src/ui/assets/generate_brain.py
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance

OUT = Path(__file__).parent / "brain_top.png"
SIZE = 600

rng = np.random.default_rng(42)
px = np.arange(SIZE)
pxx, pyy = np.meshgrid(px, px)

# ── brain outline: rounded frontal lobe shape ──────────────────────────────────
# Center the brain slightly above-center (frontal lobe prominent)
cx, cy = SIZE / 2, SIZE * 0.50
rx, ry = 250, 270

def in_brain(xs, ys):
    return ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0

brain_mask = in_brain(pxx, pyy)

# ── radial distance for lighting ──────────────────────────────────────────────
dist_norm = np.sqrt(((pxx - cx) / rx) ** 2 + ((pyy - cy) / ry) ** 2)
dist_norm_clipped = np.clip(dist_norm, 0, 1)

# Bright center (gyri highlights), darker edges
brightness = (230 - 140 * dist_norm_clipped ** 0.45).astype(np.float32)

# ── multi-octave noise for gyri/sulci surface texture ────────────────────────
def octave_noise(shape, scale, octaves=5, persistence=0.5, seed=0):
    rng_loc = np.random.default_rng(seed)
    result = np.zeros(shape)
    amp, freq = 1.0, 1.0
    for _ in range(octaves):
        n = rng_loc.standard_normal(shape)
        n = gaussian_filter(n, sigma=scale / freq)
        result += amp * n
        amp *= persistence
        freq *= 2
    return result

noise = octave_noise((SIZE, SIZE), scale=20, octaves=6, persistence=0.55, seed=7)
noise = (noise - noise.min()) / (noise.max() - noise.min())

texture = brightness - 22 * (noise - 0.5)
texture = np.clip(texture, 80, 240)

# ── interhemispheric fissure (vertical center groove) ────────────────────────
fissure_w = 6
fissure_mask = np.abs(pxx - cx) < fissure_w
fissure_t = np.clip(1 - np.abs(pxx - cx) / fissure_w, 0, 1)
texture = np.where(fissure_mask & brain_mask,
                   texture * (0.25 + 0.30 * (1 - fissure_t)),
                   texture)

# ── real brain pinkish-tan colour (frontal lobe cortex) ──────────────────────
r_ch = np.clip(texture * 1.00, 0, 255).astype(np.uint8)
g_ch = np.clip(texture * 0.82, 0, 255).astype(np.uint8)
b_ch = np.clip(texture * 0.72, 0, 255).astype(np.uint8)
alpha = np.where(brain_mask, 255, 0).astype(np.uint8)

img_arr = np.stack([r_ch, g_ch, b_ch, alpha], axis=-1)
pil_img = Image.fromarray(img_arr, mode="RGBA")

# ── draw anatomical sulci with PIL ────────────────────────────────────────────
draw = ImageDraw.Draw(pil_img)
SULCUS  = (55, 35, 25, 235)    # dark reddish-brown
FISSURE = (30, 18, 12, 250)    # very dark central fissure


def arc_pts(acx, acy, arx, ary, a0, a1, n=80):
    angles = np.linspace(np.radians(a0), np.radians(a1), n)
    return list(zip((acx + arx * np.cos(angles)).tolist(),
                    (acy + ary * np.sin(angles)).tolist()))


def wavy_hline(y, x0, x1, amplitude=6, freq=3, n=120, color=SULCUS, width=2):
    """Horizontal wavy line (sine wave) for sulci running left-right."""
    xs = np.linspace(x0, x1, n)
    phase = np.linspace(0, freq * np.pi, n)
    ys = y + amplitude * np.sin(phase)
    pts = list(zip(xs.tolist(), ys.tolist()))
    draw.line(pts, fill=color, width=width)


# -- Interhemispheric (longitudinal) fissure — bold vertical center line
draw.line([(cx, cy - ry + 15), (cx, cy + ry - 15)], fill=FISSURE, width=6)
# subtle shadow on both sides
for dx in (-3, 3):
    draw.line([(cx + dx, cy - ry + 30), (cx + dx, cy + ry - 30)],
              fill=(60, 35, 20, 90), width=2)

# -- Superior frontal sulcus (runs horizontally across upper brain)
hw = 195   # half-width from center edge to near fissure
wavy_hline(cy - 100, cx - hw, cx - 10, amplitude=7, freq=2.5, width=3)
wavy_hline(cy - 100, cx + 10, cx + hw, amplitude=7, freq=2.5, width=3)

# -- Middle frontal sulcus
wavy_hline(cy - 28, cx - hw + 10, cx - 10, amplitude=6, freq=2.8, width=2)
wavy_hline(cy - 28, cx + 10, cx + hw - 10, amplitude=6, freq=2.8, width=2)

# -- Inferior frontal sulcus
wavy_hline(cy + 48, cx - hw + 25, cx - 10, amplitude=5, freq=2.2, width=2)
wavy_hline(cy + 48, cx + 10, cx + hw - 25, amplitude=5, freq=2.2, width=2)

# -- Orbital sulcus (lower, shorter)
wavy_hline(cy + 125, cx - hw + 60, cx - 10, amplitude=4, freq=1.8, width=2)
wavy_hline(cy + 125, cx + 10, cx + hw - 60, amplitude=4, freq=1.8, width=2)

# -- Cingulate sulcus (near midline, thin)
for sign in (-1, 1):
    pts = arc_pts(cx + sign * 42, cy - 15, 12, 215, -78, 55)
    draw.line(pts, fill=(70, 42, 32, 190), width=2)

# -- Lateral (Sylvian) fissure — diagonal from lateral edge inward
for sign in (-1, 1):
    pts = arc_pts(cx + sign * 160, cy + 60, 80, 45,
                  120 if sign == -1 else 60,
                  190 if sign == -1 else -10)
    draw.line(pts, fill=(50, 28, 18, 230), width=3)

# ── outer brain outline ───────────────────────────────────────────────────────
bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
draw.ellipse(bbox, outline=(100, 65, 50, 255), width=4)

# ── specular highlight (top-left light source) ───────────────────────────────
highlight = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
hd = ImageDraw.Draw(highlight)
# soft white blob upper-left of each hemisphere
for sign in (-1, 1):
    hx = int(cx + sign * 90)
    hy = int(cy - 130)
    for r in range(55, 0, -5):
        a = int(18 * (1 - r / 55) ** 1.5)
        hd.ellipse([hx - r, hy - r, hx + r, hy + r], fill=(255, 240, 230, a))
highlight = highlight.filter(ImageFilter.GaussianBlur(radius=12))
pil_img = Image.alpha_composite(pil_img, highlight)

# ── edge vignette ─────────────────────────────────────────────────────────────
vignette = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
vd = ImageDraw.Draw(vignette)
for i in range(35):
    t = i / 35
    a = int(70 * t)
    pad = i * 3
    vd.ellipse([cx - rx + pad, cy - ry + pad, cx + rx - pad, cy + ry - pad],
               outline=(0, 0, 0, a), width=2)
pil_img = Image.alpha_composite(pil_img, vignette)

# ── slight smoothing ──────────────────────────────────────────────────────────
pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))

# ── save ──────────────────────────────────────────────────────────────────────
pil_img.save(OUT, format="PNG")
print(f"Saved {OUT}  ({SIZE}x{SIZE} px)")
