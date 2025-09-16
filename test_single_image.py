import tensorflow as tf
import numpy as np
import sys


MODEL_PATH = 'D:/Coolyeah/MAGANG/Image Processing/models/best.pt'
LABELS_PATH = 'labels.txt'
IMAGE_SIZE = (224, 224)


try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print("✅ Model dan label berhasil dimuat.")
except Exception as e:
    print(f"❌ Error memuat model atau label: {e}")
    sys.exit()

if len(sys.argv) < 2:
    print("Penggunaan: python test_single_image.py <path_ke_gambar>")
    sys.exit()
image_path = sys.argv[1]


try:

    img = tf.keras.utils.load_img(
        image_path, target_size=IMAGE_SIZE
    )

    img_array = tf.keras.utils.img_to_array(img)

    img_array = tf.expand_dims(img_array, 0) # Membuat batch

    print(f"🖼️  Gambar '{image_path}' berhasil diproses.")
except Exception as e:
    print(f"❌ Error memproses gambar: {e}")
    sys.exit()



predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0]) 


predicted_index = np.argmax(score)
predicted_label = labels[predicted_index]
confidence = 100 * np.max(score)

print("\n--- Hasil Prediksi ---")
print(f"💰 Prediksi: Uang Rp {predicted_label}")
print(f"🎯 Keyakinan: {confidence:.2f}%")
print("----------------------\n")