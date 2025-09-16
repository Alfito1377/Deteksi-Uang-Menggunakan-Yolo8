import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import io

@st.cache_resource
def load_model_and_labels():
    try:
        model = tf.keras.models.load_model('model_uang.h5')
        with open('labels.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return model, labels
    except Exception as e:
        st.error(f"Error: Gagal memuat model atau file label. Pastikan file 'model_uang.h5' dan 'labels.txt' ada.")
        st.error(e)
        return None, None

model, labels = load_model_and_labels()

def preprocess_image(image):

    img_array = np.array(image.convert('RGB'))


    resized_img = cv2.resize(img_array, (224, 224))


    processed_img = np.expand_dims(resized_img / 255.0, axis=0)
    return processed_img


st.title("💵 Aplikasi Deteksi Nominal Uang Kertas")
st.write("Gunakan kamera atau unggah gambar uang kertas untuk dideteksi nominalnya.")
st.write("Aplikasi ini dibuat sebagai contoh penggunaan TensorFlow dan Streamlit.")


tab1, tab2 = st.tabs(["Kamera (Ambil Foto)", "Unggah Gambar"])

with tab1:
    st.subheader("Ambil Foto dari Kamera")
    camera_input = st.camera_input("Arahkan kamera ke uang kertas dan ambil foto")

    if camera_input is not None and model is not None:

        image = Image.open(io.BytesIO(camera_input.read()))
        st.image(image, caption='Foto dari Kamera', use_container_width=True)
        st.write("Menganalisis foto...")


        processed_image = preprocess_image(image)


        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = labels[predicted_class_index]
        confidence = np.max(prediction) * 100


        st.subheader("Hasil Deteksi:")
        st.success(f"Ini adalah uang: **Rp {predicted_class_label}**")
        st.info(f"Tingkat Keyakinan: **{confidence:.2f}%**")

with tab2:
    st.subheader("Unggah Gambar")
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang Diunggah', use_container_width=True)
        st.write("")
        st.write("Menganalisis gambar...")

        processed_image = preprocess_image(image)


        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = labels[predicted_class_index]
        confidence = np.max(prediction) * 100


        st.subheader("Hasil Deteksi:")
        st.success(f"Ini adalah uang: **Rp {predicted_class_label}**")
        st.info(f"Tingkat Keyakinan: **{confidence:.2f}%**")