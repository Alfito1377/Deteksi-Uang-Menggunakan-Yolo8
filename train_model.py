import tensorflow as tf
import os


DATASET_DIR = 'dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# --- 1. Memuat Dataset ---
print("Memuat dataset...")
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_dataset.class_names
print("Kelas yang ditemukan (dari nama folder):", class_names)
with open('labels.txt', 'w') as f:
    for item in class_names:
        f.write("%s\n" % item)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)


print("Membangun model Transfer Learning dengan Data Augmentation...")


base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE + (3,),
                                             include_top=False,
                                             weights='imagenet')

base_model.trainable = False


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
x = data_augmentation(inputs)                                       
x = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)(x)         
x = base_model(x, training=False)                                   
x = tf.keras.layers.GlobalAveragePooling2D()(x)                     
x = tf.keras.layers.Dropout(0.2)(x)                                 
outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x) 
model = tf.keras.Model(inputs, outputs)



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

print("\n--- Memulai Pelatihan Model ---")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)
print("--- Pelatihan Model Selesai ---\n")

model.save('model_uang.h5')
print("Model final telah disimpan sebagai model_uang.h5")