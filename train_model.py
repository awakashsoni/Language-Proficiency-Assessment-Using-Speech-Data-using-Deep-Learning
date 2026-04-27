import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------- FAST SETTINGS 
tf.get_logger().setLevel('ERROR')

# -------------------- BETTER DUMMY DATA --------------------
# Simulate structured data instead of random noise
X = np.random.rand(500, 40)

# Create patterns (this improves "fake accuracy")
y_raw = np.sum(X, axis=1)
y = np.digitize(y_raw, bins=[13, 20])  # 3 classes
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Normalize
X = (X - np.mean(X)) / np.std(X)

# -------------------- MODEL --------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(40,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

# -------------------- COMPILE --------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------- FAST TRAINING --------------------
early_stop = EarlyStopping(
    monitor='loss',
    patience=2,
    restore_best_weights=True
)

model.fit(
    X, y,
    epochs=20,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -------------------- SAVE SMALL + FAST MODEL --------------------
model.save("model.h5", include_optimizer=False)

print("✅ Optimized model saved as model.h5")