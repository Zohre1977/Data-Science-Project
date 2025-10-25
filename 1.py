import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


dataset_dir = "flowers"  

data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),tf.keras.layers.RandomZoom(0.1),])
num_augmented_per_image = 5  
img_height, img_width = 224, 224
dataset_dir = "flowers" 
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    for fname in os.listdir(class_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img_path = os.path.join(class_path, fname)
            img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) 
            
            for i in range(num_augmented_per_image):
                augmented_img = data_augmentation(img_array)
                augmented_img = tf.keras.preprocessing.image.array_to_img(augmented_img[0])                
             
                new_fname = f"{os.path.splitext(fname)[0]}_aug_{i}.jpg"
                augmented_img.save(os.path.join(class_path, new_fname))
print("Augmentation and saving done!")

img_height, img_width = 224, 224
X = []
y = []
class_names = sorted(os.listdir(dataset_dir))
class_to_label = {name: idx for idx, name in enumerate(class_names)}
for class_name in class_names:
    class_path = os.path.join(dataset_dir, class_name)
    for fname in os.listdir(class_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img_path = os.path.join(class_path, fname)
            img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            X.append(img_array)
            y.append(class_to_label[class_name])
X = np.array(X, dtype='float32') / 255.0 
y = np.array(y)
print(f"Total images: {len(X)}, Classes: {class_names}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal"),layers.RandomRotation(0.1),])

base_model = MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
base_model.trainable = False
model = models.Sequential([layers.Input(shape=(224,224,3)),data_augmentation,base_model,
                           layers.GlobalAveragePooling2D(),
                           layers.Dense(102, activation='softmax')])  

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs=10,batch_size=32)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save("flower_classifier_model.h5")
print("Model saved successfully!")


