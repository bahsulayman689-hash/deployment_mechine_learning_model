import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model
from keras.layers import Layer, Dense


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape)
print(y_train_full.shape)
print(X_test.shape)
print(y_test.shape)
print(X_train_full.shape)

X_valid , X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# =====================================================================
# 🎛️ CRITICAL FIX: RESHAPE ARRAYS TO 4D TO ALIGN PERFECTLY WITH CONV2D
# =====================================================================
X_train = X_train.reshape(-1, 28, 28, 1)
X_valid = X_valid.reshape(-1, 28, 28, 1)
X_test = (X_test / 255.0).reshape(-1, 28, 28, 1)
y_test = y_test



print(X_train)
print(X_valid)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(class_names[y_train[0]])

plt.imshow(X_train[0].reshape(28, 28))
plt.show()

print(y_train[0])
print(y_train.shape)

# 4. BUILD THE CONV2D MODEL ARCHITECTURE
model = Sequential()

# First Convolutional Layer: Extracts basic edges and curves
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) # Downsamples the image features
model.add(keras.layers.Dropout(0.25))

# Second Convolutional Layer: Extracts complex combinations of shapes
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

# Flattening and feeding features into dense classification layers
model.add(keras.layers.Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(Dense(units=10, activation="softmax"))


print(model.summary())
print(model.layers)
hi = model.layers[1]
print(hi.name)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])
def exp_decay(lr0, s):
    def exp_decay_fn(epoch):
        return lr0 * 0.01 ** (epoch/s)
    return exp_decay_fn
exponential_decay_fn= exp_decay(lr0=0.01, s=20)
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

checkpoint_cb = keras.callbacks.ModelCheckpoint("fashion_mnist_data.h5",
                                                mode='min',
                                                save_best_only=True,
                                                initial_value_threshold=None,
                                                verbose=1,
                                                monitor='val_loss')
log_csv_cb = keras.callbacks.CSVLogger(filename="loss.csv")
history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_valid, y_valid),
                    batch_size=32,
                    callbacks=[lr_scheduler, checkpoint_cb, log_csv_cb])
model.save("fashion_mnist_data.h5")
print(history.params)
print(history.epoch)
print(history.history.keys())
pd.DataFrame(history.history).plot(figsize=(8, 6))
plt.gca().set_ylim(0, 1)
plt.title("model accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(["accuracy", 'val_accuracy'], loc="lower right")
plt.show()
acc, val_loss = model.evaluate(X_test, y_test, batch_size=32)
print(f"Accuracy of model is {acc}")
print(f"Val_Loss {val_loss}")
best_model = load_model("fashion_mnist_data.h5")
# =====================================================================
# ADD THIS TO THE BOTTOM OF YOUR MAIN TRAINING SCRIPT TO PLOT AND CHECK
# =====================================================================
from PIL import Image, ImageOps

# 1. Provide the exact path to your uploaded shirt image file
image_filename = "fashion_mnist_2.jpg"

try:
    # Open and process the image file matching your app pipeline
    raw_img = Image.open(image_filename)
    raw_img = ImageOps.exif_transpose(raw_img)
    gray_img = raw_img.convert("L")
    resized_img = gray_img.resize((28, 28))
    
    # Convert to array and scale down to 0.0 - 1.0
    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    
    # Auto-invert background contrast if it's too bright 
    # (Forces a dark background matching Fashion MNIST specifications)
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array
        
    # Shape into model input format: (1, 28, 28)
    model_input = img_array.reshape(1, 28, 28, 1)
    
    # Run prediction pass
    raw_predictions = model.predict(model_input, verbose=0)[0]
    predicted_index = np.argmax(raw_predictions)
    model_confidence = raw_predictions[predicted_index]
    predicted_label = class_names[predicted_index]
    
    # 2. PLOT THE MODEL'S VIEW AND DISPLAY PREDICTION
    plt.figure(figsize=(5, 5))
    
    # We display the processed 28x28 pixel array passed into the model
    plt.imshow(img_array, cmap="gray")
    plt.axis("off")
    
    # Title display showing what the model decided
    plt.title(
        f"🤖 Model Prediction Matrix\n"
        f"Result: {predicted_label}\n"
        f"Confidence: {model_confidence:.2%}",
        fontsize=12,
        fontweight="bold",
        pad=10
    )
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"\n❌ Error: Could not find '{image_filename}' in your folder path.")
    print("💡 Please save your shirt picture in the same directory as this training script.\n")
