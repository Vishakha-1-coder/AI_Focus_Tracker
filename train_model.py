import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Path and Config ---
DATA_DIR = "dataset/imgs/train"
IMG_SIZE = 100

X = []
y = []

# --- Load Data and Assign Labels ---
for folder in os.listdir(DATA_DIR):
    label = 0 if folder == "c0" else 1  # 0 = focused, 1 = distracted
    folder_path = os.path.join(DATA_DIR, folder)

    for img_name in os.listdir(folder_path)[:500]:  # use 500 images per class
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

X = np.array(X) / 255.0
y = to_categorical(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Data loaded:", len(X_train), "training and", len(X_val), "validation images")

# --- Build CNN ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train ---
es = EarlyStopping(patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[es])

# --- Save Model ---
model.save("model/model.h5")
print("✅ Model saved as model/model.h5")

