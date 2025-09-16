import cv2
import numpy as np
import tensorflow as tf
import time


MODEL_PATH = 'model_uang.h5'
LABELS_PATH = 'labels.txt'
IMAGE_SIZE = (224, 224) 


try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"Model {MODEL_PATH} dan label {LABELS_PATH} berhasil dimuat.")
except Exception as e:
    print(f"Error: Gagal memuat model atau file label. Pastikan file '{MODEL_PATH}' dan '{LABELS_PATH}' ada.")
    print(e)
    exit()


cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("Error: Tidak dapat mengakses kamera. Pastikan kamera terhubung dan tidak digunakan oleh aplikasi lain.")
    exit()

print("Kamera berhasil diinisialisasi. Tekan 'q' untuk keluar.")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal mengambil frame dari kamera. Keluar...")
        break

   
    frame = cv2.flip(frame, 1)


    img_for_prediction = cv2.resize(frame, IMAGE_SIZE)


    img_for_prediction = cv2.cvtColor(img_for_prediction, cv2.COLOR_BGR2RGB)


    img_for_prediction = np.expand_dims(img_for_prediction / 255.0, axis=0)


    predictions = model.predict(img_for_prediction, verbose=0)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100


    predicted_label = labels[predicted_class_index]
    text = f"Prediksi: Rp {predicted_label} ({confidence:.2f}%)"


    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow('Deteksi Uang Realtime', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows() 
print("Aplikasi deteksi realtime dihentikan.")