# Google Drive Model Deployment Guide

## 📦 How to Upload Models to Google Drive and Deploy to Render

### Step 1: Upload Models to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Upload these 3 files:
   - `yolo11x_leaf.pt`
   - `plant_disease_model_optimized.h5`
   - `plant_disease_model_optimized.keras`

### Step 2: Get Shareable Links

For **each file**:
1. Right-click on the file → **Share** or **Get link**
2. Change access to: **"Anyone with the link"**
3. Copy the link - it looks like:
   ```
   https://drive.google.com/file/d/1ABC123xyz456DEF/view?usp=sharing
   ```
4. Extract the **FILE_ID** (the part between `/d/` and `/view`)
   - Example: `1ABC123xyz456DEF`

### Step 3: Configure download_models.py

Open `download_models.py` and replace these lines:

```python
MODELS = {
    'yolo11x_leaf.pt': {
        'file_id': 'YOUR_YOLO_FILE_ID_HERE',  # ← Paste your YOLO file ID
        'size_mb': 200
    },
    'plant_disease_model_optimized.h5': {
        'file_id': 'YOUR_MOBILENET_H5_FILE_ID_HERE',  # ← Paste your H5 file ID
        'size_mb': 30
    },
    'plant_disease_model_optimized.keras': {
        'file_id': 'YOUR_MOBILENET_KERAS_FILE_ID_HERE',  # ← Paste your KERAS file ID
        'size_mb': 30
    }
}
```

### Step 4: Deploy to Render

Now your models will be **downloaded automatically** on Render startup!

```bash
git add .
git commit -m "Deploy with Google Drive models"
git push origin main
```

## 🎯 How It Works

1. When Render starts your app, `app.py` calls `download_models()`
2. `download_models.py` checks if model files exist locally
3. If not, it downloads them from Google Drive
4. Models are cached on Render's filesystem
5. App starts normally with downloaded models

## ⚠️ Important Notes

- **First deployment will be SLOW** (~5-10 min) while downloading models
- Models persist between deploys (unless you change plans)
- Free tier has **512MB RAM** - may still crash with all 3 models loaded
- Consider using **only the necessary model** (.h5 or .keras, not both)

## 🚀 Alternative: Use Only One MobileNet Model

If RAM is tight, keep only `plant_disease_model_optimized.h5` and remove the `.keras` file.
