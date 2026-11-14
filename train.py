# LeafWatch - training + research experiments. Run in Google Colab (GPU runtime).
# Dataset: PlantVillage (Kaggle "plantvillage dataset"), folder with class sub-folders.
# Output: leaf_model.keras, class_names.json  -> put both next to app.py
import os, json, numpy as np, cv2, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier

DATA = "PlantVillage"          # <- change to your dataset folder path
KEEP = ("Tomato", "Potato")    # crops used in the prototype
IMG, EPOCHS = 224, 5

classes = sorted(c for c in os.listdir(DATA) if c.startswith(KEEP))
print("Classes:", classes)
kw = dict(class_names=classes, image_size=(IMG, IMG), batch_size=32,
          seed=42, validation_split=0.2)
train_raw = keras.utils.image_dataset_from_directory(DATA, subset="training", **kw)
val_raw = keras.utils.image_dataset_from_directory(DATA, subset="validation", **kw)

prep = lambda x, y: (keras.applications.mobilenet_v2.preprocess_input(x), y)
aug = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"),
                        layers.RandomRotation(0.1),
                        layers.RandomBrightness(0.2),   # helps real-world lighting
                        layers.RandomContrast(0.2)])
AT = tf.data.AUTOTUNE
train = train_raw.map(lambda x, y: (aug(x, training=True), y)).map(prep).prefetch(AT)
val = val_raw.map(prep).prefetch(AT)

# ---- Model: MobileNetV2 transfer learning (base.input kept so Grad-CAM works)
base = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3),
                                      include_top=False, weights="imagenet")
base.trainable = False
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.2)(x)
out = layers.Dense(len(classes), activation="softmax")(x)
model = keras.Model(base.input, out)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=EPOCHS)

model.save("leaf_model.keras")
json.dump(classes, open("class_names.json", "w"))
cnn_acc = model.evaluate(val)[1]
print(f"\n[Experiment 1] CNN accuracy on dataset validation set: {cnn_acc:.3f}")

# ---- Experiment 2: robustness to lighting (simulated dark / bright photos)
for factor in (0.5, 1.5):
    ds = val_raw.map(lambda x, y: (tf.clip_by_value(x * factor, 0, 255), y)).map(prep)
    print(f"[Experiment 2] accuracy with brightness x{factor}: {model.evaluate(ds, verbose=0)[1]:.3f}")

# ---- Experiment 3: colour-only model vs CNN (which cues matter?)
X, y = [], []
for xb, yb in val_raw.take(40):
    for img, lab in zip(xb.numpy().astype("uint8"), yb.numpy()):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        X.append(h / h.sum()); y.append(lab)
X, y = np.array(X), np.array(y)
cut = len(X) // 2
rf = RandomForestClassifier(200, random_state=0).fit(X[:cut], y[:cut])
print(f"[Experiment 3] colour-histogram-only model accuracy: {rf.score(X[cut:], y[cut:]):.3f} "
      f"(vs CNN {cnn_acc:.3f})")
print("\nExperiment 4 (phone-photo generalisation) is done inside the app's 'Research' tab.")
