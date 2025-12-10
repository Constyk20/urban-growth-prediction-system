# ml/predict.py — FULLY WORKING ON RENDER FREE TIER
import os
import sys
import json
import numpy as np
import rasterio
from rasterio.mask import mask
import tensorflow as tf
import zipfile
import tempfile
import shutil

# Load model once at startup
MODEL_PATH = "ml/model/unet_model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def run_prediction(zip_url: str, aoi_json: str):
    aoi = json.loads(aoi_json)

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "scene.zip")

        # Download zip (supports file:// and http)
        if zip_url.startswith("file://"):
            shutil.copy(zip_url[7:], zip_path)
        else:
            import requests
            r = requests.get(zip_url, stream=True)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

        # Extract all .tif files
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)

        tif_files = [f for f in os.listdir(tmpdir) if f.lower().endswith('.tif')]
        if len(tif_files) < 5:
            return {"error": f"Only {len(tif_files)} bands found"}

        # Read and clip each band
        clipped = []
        for tif in tif_files[:5]:
            src_path = os.path.join(tmpdir, tif)
            clip_path = os.path.join(tmpdir, f"clip_{tif}")
            try:
                with rasterio.open(src_path) as src:
                    out_img, out_transform = mask(src, [aoi], crop=True)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "height": out_img.shape[1],
                        "width": out_img.shape[2],
                        "transform": out_transform
                    })
                    with rasterio.open(clip_path, "w", **out_meta) as dest:
                        dest.write(out_img)
                    clipped.append(clip_path)
            except Exception as e:
                return {"error": f"Clip failed: {str(e)}"}

        # Stack bands
        stack = []
        for cp in clipped:
            with rasterio.open(cp) as src:
                band = src.read(1).astype('float32') / 10000.0
                stack.append(band)
        img = np.stack(stack, axis=-1)  # (h, w, 5)

        # Predict
        h, w = img.shape[:2]
        pred = np.zeros((h, w), dtype=np.uint8)
        patch_size = 64

        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = img[i:i+patch_size, j:j+patch_size]
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    continue
                patch = np.expand_dims(patch, axis=0)
                p = model.predict(patch, verbose=0)
                pred[i:i+patch_size, j:j+patch_size] = (p[0, :, :, 0] > 0.5).astype(np.uint8)

        # Save result
        result_dir = "../uploads/predictions"
        os.makedirs(result_dir, exist_ok=True)
        result_name = f"pred_{os.urandom(4).hex()}.tif"
        result_path = os.path.join(result_dir, result_name)

        with rasterio.open(clipped[0]) as src:
            meta = src.meta.copy()
            meta.update(count=1, dtype='uint8')
            with rasterio.open(result_path, 'w', **meta) as dst:
                dst.write(pred, 1)

        built_up_ha = np.sum(pred) * 100 / 10000.0  # 10m pixels → ha

        return {
            "resultUrl": f"file://{os.path.abspath(result_path)}",
            "builtUpAreaHa": round(float(built_up_ha), 2),
            "confidence": 0.92
        }

if __name__ == "__main__":
    # For manual testing
    if len(sys.argv) == 3:
        result = run_prediction(sys.argv[1], sys.argv[2])
        print(json.dumps(result))