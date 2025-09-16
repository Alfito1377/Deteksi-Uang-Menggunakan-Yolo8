# Deteksi Mata Uang Real-time (YOLOv8 & Streamlit)

Proyek ini adalah aplikasi web yang dibuat dengan Streamlit untuk melakukan deteksi objek pada mata uang (contoh: Rupiah) secara *real-time* menggunakan webcam. Aplikasi ini ditenagai oleh model *object detection* **YOLOv8**.

Selain deteksi *real-time*, proyek ini juga menyertakan *script* terpisah untuk menguji model pada satu gambar.



---

## 🚀 Fitur

* **Deteksi Real-time:** Menggunakan webcam untuk mendeteksi mata uang secara langsung.
* **Deteksi Gambar Tunggal:** Kemampuan untuk menguji performa model pada gambar statis.
* **Interface Web:** Dibangun menggunakan Streamlit untuk antarmuka yang ramah pengguna.
* **Powered by YOLOv8:** Menggunakan model YOLOv8 dari Ultralytics untuk deteksi objek yang cepat dan akurat.

---

---

## 🔧 Instalasi

Untuk menjalankan proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

Jalankan "pip install -r requirements.txt "

ධ Cara Penggunaan
Ada dua cara untuk menjalankan proyek ini:

1. Menjalankan Aplikasi Web (Streamlit)
Gunakan perintah ini di terminal Anda:

Bash

streamlit run app.py
Setelah itu, buka browser Anda dan akses alamat URL lokal yang ditampilkan (biasanya http://localhost:8501).

2. Tes pada Gambar Tunggal
Untuk menguji deteksi pada satu gambar, gunakan script test_single_image.py diikuti dengan path ke gambar yang ingin Anda uji.

Bash

python test_single_image.py images/test_100k.jpeg
(Pastikan path images/test_100k.jpeg sesuai dengan lokasi gambar uji Anda)

🛠️ Teknologi yang Digunakan
Python

YOLOv8 (Ultralytics) - Untuk model deteksi objek.

Streamlit - Untuk membangun interface aplikasi web.

OpenCV - Untuk pemrosesan gambar dan video.
