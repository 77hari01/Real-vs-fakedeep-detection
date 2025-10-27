from google.colab import drive
drive.mount('/content/drive')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
train_path = '/content/drive/MyDrive/7030/train'
valid_path = '/content/drive/MyDrive/7030/valid'
test_path  = '/content/drive/MyDrive/7030/test'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
)
valid_gen = ImageDataGenerator(rescale=1./255)
test_gen  = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
val_ds   = valid_gen.flow_from_directory(valid_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
test_ds  = test_gen.flow_from_directory(test_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

base_model = Xception(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

test_loss, test_acc = model.evaluate(test_ds)
print("✅ Test Accuracy:", test_acc)

for layer in base_model.layers[-30:]:  # last 30 layers
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=5)

model.save_weights('/content/drive/MyDrive/xceptionnew.weights.h5')
# Load weights into the existing model
model.load_weights('/content/drive/MyDrive/xceptionnew.weights.h5')
print("Weights loaded successfully!")

#testing
def load_and_preprocess(img_path):
    from tensorflow.keras.preprocessing import image
    import numpy as np
    import tensorflow as tf

    # Resize to 224x224 (your model input size)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Use the same preprocessing you used during training
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    return img_array

img_path = '/content/real_20.jpg'
img_array = load_and_preprocess(img_path)

pred = model.predict(img_array)
print("Raw prediction:", pred)

label = "Real Image" if pred[0][0] > 0.5 else "Deepfake Image"
print("Predicted label:", label)
