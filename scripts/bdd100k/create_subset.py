import os
import shutil
import pandas as pd

CSV_PATH = "data/raw/labels/train.csv"
IMAGE_DIR = "data/raw/bdd100k/bdd100k/images/100k/train"   # adapt this
OUT_IMG_DIR = "data/samples/test_images"
OUT_META_PATH = "data/samples/metadata/sample_train.csv"

N = 20
SEED = 42

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs("data/samples/metadata", exist_ok=True)

df = pd.read_csv(CSV_PATH)

image_col = "name"

sample_df = df.sample(n=min(N, len(df)), random_state=SEED).copy()

kept_rows = []

for _, row in sample_df.iterrows():
    img_name = row[image_col]

    src = os.path.join(IMAGE_DIR, img_name)
    dst = os.path.join(OUT_IMG_DIR, os.path.basename(img_name))

    if os.path.exists(src):
        shutil.copy2(src, dst)
        kept_rows.append(row)
    else:
        print("Missing:", src)

pd.DataFrame(kept_rows).to_csv(OUT_META_PATH, index=False)

print(f"Subset saved to {OUT_META_PATH}")
print(f"Copied {len(kept_rows)} images to {OUT_IMG_DIR}")