import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === DATASET ===
train_dir = "dataset"
img_size = (128, 128)
batch_size = 16

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=batch_size,
    subset='training'
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=batch_size,
    subset='validation'
)

# === MODEL ===
model = models.Sequential([
    layers.Conv2D(64, (3,3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes: H, S, U
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("letter_model_v2.h5")
print("Model saved as letter_model_v2.h5")

