
import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

# ====== Konfigurasi ======
MODEL_PATH = "../Datasets/detect_Plastic.h5"
CSV_PATH = "../Datasets/_annotations.csv"
IMG_DIR = "../Datasets/images"
IOU_THRESHOLD = 0.5

# ====== Fungsi Hitung IoU ======
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# ====== Muat Model ======
model = load_model(MODEL_PATH)

# ====== Muat Data Anotasi ======
df = pd.read_csv(CSV_PATH)

# ====== Evaluasi ======
ious = []
y_true = []
y_pred = []

image_names = df['plastic-12-_jpg.rf.3b58a8ec37f1817b91ed056b4bf7a0ef.jpg'].unique()

for img_name in image_names:
    img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(img_path):
        continue

    # Ambil ground truth box (asumsi 1 box per gambar)
    gt_data = df[df['plastic-12-_jpg.rf.3b58a8ec37f1817b91ed056b4bf7a0ef.jpg'] == img_name].iloc[0]
    gt_box = [gt_data['xmin'], gt_data['ymin'], gt_data['xmax'], gt_data['ymax']]

    # Baca dan preprocess gambar
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # Prediksi bounding box (hasil dalam rasio 0–1)
    pred = model.predict(img_input)[0]
    pred_box = [
        int(pred[0] * width),
        int(pred[1] * height),
        int(pred[2] * width),
        int(pred[3] * height),
    ]

    # Hitung IoU
    iou = compute_iou(gt_box, pred_box)
    ious.append(iou)

    # Klasifikasi benar/salah berdasarkan ambang batas IoU
    y_true.append(1)
    y_pred.append(1 if iou >= IOU_THRESHOLD else 0)

# ====== Statistik Evaluasi ======
mean_iou = np.mean(ious)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Mean IoU     : {mean_iou:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1-Score     : {f1:.4f}")
