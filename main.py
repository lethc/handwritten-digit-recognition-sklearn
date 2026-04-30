import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import seaborn as sns

# 1. Cargar dataset load_digits
digits = load_digits()
X = digits.images  # imágenes de 8x8 píxeles
y = digits.target  # etiquetas del 0 al 9

# 2. Normalizar dividiendo entre 255
X = X / 255.0

# 3. Agregar canal de color para capas convolucionales (8,8,1)
X = X.reshape(X.shape[0], 8, 8, 1)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Construir modelo con capas de convolución
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(8, 8, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrenar
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, verbose=1)

# 5. Predicciones totales
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# 6. Métricas
print("="*50)
print("ACCURACY TOTAL:", accuracy_score(y_test, y_pred))
print("="*50)
print("\nREPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# 7. Predicción individual con visualización
indice_muestra = 0
muestra = X_test[indice_muestra]
etiqueta_real = y_test[indice_muestra]

pred_individual = model.predict(muestra.reshape(1, 8, 8, 1))
pred_clase = np.argmax(pred_individual)

print("="*50)
print("PREDICCIÓN INDIVIDUAL:")
print(f"Etiqueta real: {etiqueta_real}")
print(f"Predicción del modelo: {pred_clase}")
print(f"Probabilidades: {pred_individual[0]}")

# 8. Visualización en pyplot - Comparación de predicciones vs reales
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flat

for i in range(10):
    axes[i].imshow(X_test[i].reshape(8, 8), cmap='gray')
    axes[i].set_title(f"Real: {y_test[i]}\nPred: {y_pred[i]}")
    axes[i].axis('off')

plt.suptitle('Comparación de Predicciones vs Etiquetas Reales', fontsize=14)
plt.tight_layout()
plt.show()

# Visualización adicional: la predicción individual
plt.figure(figsize=(4, 4))
plt.imshow(muestra.reshape(8, 8), cmap='gray')
plt.title(f"PREDICCIÓN INDIVIDUAL\nReal: {etiqueta_real} → Predicho: {pred_clase}")
plt.axis('off')
plt.show()
