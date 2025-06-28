import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from keras.models import load_model  # ← pakai ini

@st.cache_resource
def load_model_vgg():
    return load_model("detect_plastic.h5")

model = load_model_vgg()

# UI
st.title("Deteksi Sampah Plastik dengan VGG16")
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka dan tampilkan gambar asli
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Asli", use_column_width=True)

    # Preprocessing: Resize sesuai input model
    img_resized = image.resize((224, 224))  # Sesuaikan dengan input model VGG16 kamu
    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)  # [1, 224, 224, 3]

    # Prediksi
    with st.spinner("Mendeteksi objek..."):
        prediction = model.predict(img_input)[0]  # asumsi: output = [xmin, ymin, xmax, ymax]

        # Skala kembali ke ukuran gambar asli
        width, height = image.size
        xmin = int(prediction[0] * width)
        ymin = int(prediction[1] * height)
        xmax = int(prediction[2] * width)
        ymax = int(prediction[3] * height)

        # Gambar bounding box
        draw = ImageDraw.Draw(image)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        st.image(image, caption="Hasil Deteksi", use_column_width=True)
