# tensorFlow.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import collections

DATA_DIR = "imagenes_fotos/"
img_height = 224
img_width = 224
batch_size = 32
epochs = 20

# CARGAR Y VERIFICAR DATOS
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
class_counts = collections.Counter([label.numpy().item() for batch, labels in train_ds for label in labels])

# MODELO
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet')
base_model.trainable = True

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping])
model.save("modelo_clasificador.h5")

#GRÁFICAS DE ENTRENAMIENTO
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Accuracy durante entrenamiento')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Loss durante entrenamiento')
plt.legend()
plt.show()

# INTERFAZ GRÁFICA
class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        self.configurar_ventana()
        self.cargar_modelo()
        self.construir_interfaz()

    def configurar_ventana(self):
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.master.title("Clasificador Inteligente")
        self.master.geometry("500x650")
        self.master.resizable(False, False)

    def cargar_modelo(self):
        self.model = tf.keras.models.load_model("modelo_clasificador_mejorado.h5")
        self.class_names = class_names

    def construir_interfaz(self):
        self.main_frame = ctk.CTkFrame(self.master, corner_radius=15)
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Clasificador de Imágenes",
            font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=(10, 20))

        self.image_label = ctk.CTkLabel(self.main_frame, text="")
        self.image_label.pack(pady=10)

        self.result_label = ctk.CTkLabel(
            self.main_frame,
            text="Selecciona una imagen",
            font=ctk.CTkFont(size=14))
        self.result_label.pack(pady=(20, 5))

        self.prob_label = ctk.CTkLabel(
            self.main_frame,
            text="",
            font=ctk.CTkFont(size=12),
            justify="left")
        self.prob_label.pack(pady=(5, 20))

        self.btn = ctk.CTkButton(
            self.main_frame,
            text="Cargar Imagen",
            command=self.cargar_y_predecir)
        self.btn.pack(pady=10)

    def cargar_y_predecir(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        try:
            img = Image.open(file_path).resize((img_width, img_height))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk

            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            self.result_label.configure(
                text=f"Predicción: {self.class_names[predicted_class]}",
                text_color="green" if confidence > 75 else "orange")

            prob_text = "\n".join(
                [f"{self.class_names[i]}: {prob * 100:.1f}%" for i, prob in enumerate(predictions[0])])
            self.prob_label.configure(text=prob_text)

        except Exception as e:
            self.result_label.configure(text=f"Error: {str(e)}", text_color="red")
            
if __name__ == "__main__":
    root = ctk.CTk()
    app = ImageClassifierApp(root)
    root.mainloop()
