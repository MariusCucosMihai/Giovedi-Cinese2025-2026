import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# === Config ===
LETTERS = ['S', 'H', 'U']
OUTPUT_DIR = "dataset"
IMG_SIZE = 128
NUM_IMAGES = 50

# === Font ===
# Prefer Arial if available
POSSIBLE_FONTS = [
    "/home/pi/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
]
FONT_PATH = next((f for f in POSSIBLE_FONTS if os.path.exists(f)), None)
if FONT_PATH is None:
    raise FileNotFoundError("⚠️ No suitable font found. Please install Arial or DejaVuSans.")

FONT_SIZE = 95

# === Create folders ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
for letter in LETTERS:
    os.makedirs(os.path.join(OUTPUT_DIR, letter), exist_ok=True)

# === Generate images ===
for letter in LETTERS:
    print(f"Generating dataset for {letter}...")
    for i in range(NUM_IMAGES):
        # White background
        img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Load font
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

        # Get text size and position
        text_bbox = draw.textbbox((0, 0), letter, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = (IMG_SIZE - text_w) // 2 + random.randint(-3, 3)
        text_y = (IMG_SIZE - text_h) // 2 + random.randint(-3, 3)

        # --- Draw thick filled letter ---
        # Draw multiple layers for bold/thickness
        for dx in [-5, 0, 5]:
            for dy in [-5, 0, 5]:
                draw.text((text_x + dx, text_y + dy), letter, font=font, fill=(1, 1, 1))

        # Apply small rotation and scaling
        angle = random.uniform(-5, 5)
        scale = random.uniform(0.9, 1.1)

        scaled_size = int(IMG_SIZE * scale)
        img = img.resize((scaled_size, scaled_size), Image.Resampling.LANCZOS)
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))

        # Center crop to original size
        img = img.crop((
            (img.width - IMG_SIZE) // 2,
            (img.height - IMG_SIZE) // 2,
            (img.width + IMG_SIZE) // 2,
            (img.height + IMG_SIZE) // 2
        ))

        # Optional slight blur and brightness
        img = img.filter(ImageFilter.SMOOTH_MORE)
        np_img = np.array(img)
        brightness = random.randint(0, 15)
        np_img = np.clip(np_img + brightness, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)

        # Save
        filename = os.path.join(OUTPUT_DIR, letter, f"{letter}_{i:03d}.png")
        img.save(filename)

print("✅ Dataset generated successfully!")
print(f"Check the '{OUTPUT_DIR}/' folder.")
