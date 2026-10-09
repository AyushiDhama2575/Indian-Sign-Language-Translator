import os
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data"

actions = np.array([
    'you',
    'good',
    'thank you',
    'me',
    'my',
    'hello'
])

no_sequences = 30
sequence_length = 30

FEATURES = 63


# ============================================================
# CREATE DATASET
# ============================================================

sequences = []
labels = []


print("\nLoading training data...\n")


for action_index, action in enumerate(actions):

    for sequence in range(no_sequences):

        window = []

        for frame_num in range(sequence_length):

            npy_path = os.path.join(
                DATA_PATH,
                action,
                str(sequence),
                str(frame_num) + ".npy"
            )

            if not os.path.exists(npy_path):

                print(
                    f"Missing: {npy_path}"
                )

                break

            res = np.load(npy_path)

            # Check 63 features
            if res.shape != (FEATURES,):

                print(
                    f"Wrong shape: {npy_path} "
                    f"-> {res.shape}"
                )

                break

            window.append(res)

        # Only complete sequences
        if len(window) == sequence_length:

            sequences.append(window)
            labels.append(action_index)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(sequences, dtype=np.float32)
y = np.array(labels)


print("========================================")
print("DATASET INFORMATION")
print("========================================")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Actions:", actions)
print("Features per frame:", FEATURES)
print("Frames per sequence:", sequence_length)


# ============================================================
# CHECK DATASET
# ============================================================

if len(X) == 0:

    raise ValueError(
        "Training data nahi mila. "
        "Pehle data collection run karo."
    )


if X.shape[1:] != (sequence_length, FEATURES):

    raise ValueError(
        f"Wrong X shape: {X.shape}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# LSTM MODEL
# ============================================================

model = Sequential([

    LSTM(
        128,
        return_sequences=True,
        activation="tanh",
        input_shape=(sequence_length, FEATURES)
    ),

    Dropout(0.2),

    LSTM(
        64,
        return_sequences=False,
        activation="tanh"
    ),

    Dropout(0.2),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        len(actions),
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)


# ============================================================
# TRAINING
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================\n")


history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# EVALUATION
# ============================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Test Loss: {loss:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = "model.h5"

model.save(MODEL_PATH)

print(
    f"\nModel saved successfully: {MODEL_PATH}"
)