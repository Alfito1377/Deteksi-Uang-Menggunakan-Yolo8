import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import numpy as np
import av
from PIL import Image


st.set_page_config(
    page_title="Deteksi Uang Rupiah",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💵 Deteksi Uang Rupiah")
st.write("Aplikasi ini menggunakan model YOLOv8 untuk mendeteksi nominal uang. Pilih mode di bawah ini.")

MODEL_PATH = 'D:/Coolyeah/MAGANG/Image Processing/models/best.pt'

@st.cache_resource
def load_yolo_model(path):
    """Memuat model YOLO dari path yang diberikan."""
    try:
        model = YOLO(path)
        return model
    except FileNotFoundError:
        st.error(f"File model tidak ditemukan di path: {path}. Pastikan Anda sudah melatih model deteksi objek (YOLO) dan meletakkannya di folder yang benar.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat model. Error: {e}")
        return None

model = load_yolo_model(MODEL_PATH)

with st.sidebar:
    st.header("Pengaturan")
    confidence_threshold = st.slider(
        "Tingkat Kepercayaan (Confidence)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.25,
        step=0.05
    )
    st.info("Slider ini berlaku untuk mode Webcam dan Unggah Gambar.")


if model is not None:
    tab1, tab2 = st.tabs(["👁️ Deteksi Real-time (Webcam)", "🖼️ Deteksi dari Gambar"])

    with tab1:
        st.header("Deteksi menggunakan Kamera")
        
        class VideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.model = model
                self.confidence_threshold = confidence_threshold

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                results = self.model.predict(source=img, conf=self.confidence_threshold, verbose=False)
                annotated_frame = results[0].plot()
                return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        webrtc_streamer(
            key="webcam-detection",
            video_transformer_factory=VideoTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )
        st.info("Klik 'START' untuk memulai kamera. Pastikan Anda memberikan izin akses kamera pada browser.")

    with tab2:
        st.header("Deteksi dari Gambar yang Diunggah")
        uploaded_file = st.file_uploader("Unggah sebuah gambar uang...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            st.image(img_array, caption="Gambar Asli", use_container_width=True)
            st.divider()

            results = model.predict(source=img_array, conf=confidence_threshold)
            annotated_image = results[0].plot()

            st.image(annotated_image, caption="Hasil Deteksi", use_container_width=True)

            if results[0].boxes:
                st.success(f"Berhasil mendeteksi {len(results[0].boxes)} objek!")
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    st.write(f"- **{class_name}** dengan tingkat kepercayaan {confidence:.2f}")
            else:
                st.warning("Tidak ada objek yang terdeteksi pada gambar ini dengan tingkat kepercayaan yang diatur.")

else:
    st.warning("Model tidak dapat dimuat. Aplikasi tidak dapat berjalan.")