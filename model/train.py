import numpy as np
import os
import pickle
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# === Cargar datos
data = np.load('data/mfcc_custom_dataset.npz')
X = data['mfccs']
y = data['labels']

# === Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# === Dividir datos
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Optuna: función objetivo
def objective(trial):
    n_units_1 = trial.suggest_int("n_units_1", 64, 512)
    n_units_2 = trial.suggest_int("n_units_2", 32, 256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])

    model = Sequential()
    model.add(Dense(n_units_1, activation='relu', input_shape=(X.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_units_2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(y_categorical.shape[1], activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=50,
              batch_size=batch_size,
              verbose=0,
              callbacks=[early_stop])

    y_pred = model.predict(X_val, verbose=0)
    acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))
    return acc

# === Ejecutar Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)

# === Entrenar modelo final con mejores parámetros
best_params = study.best_params
print("🔧 Mejores hiperparámetros:", best_params)

final_model = Sequential()
final_model.add(Dense(best_params['n_units_1'], activation='relu', input_shape=(X.shape[1],)))
final_model.add(BatchNormalization())
final_model.add(Dropout(best_params['dropout_rate']))
final_model.add(Dense(best_params['n_units_2'], activation='relu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(best_params['dropout_rate']))
final_model.add(Dense(y_categorical.shape[1], activation='softmax'))

final_model.compile(optimizer=best_params['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=best_params['batch_size'],
    callbacks=[early_stop],
    verbose=1
)

# === Guardar modelo y codificador
os.makedirs('model', exist_ok=True)
final_model.save('model/model_custom.h5')
with open('model/label_encoder_custom.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Entrenamiento final completo. Modelo y codificador guardados.")
# === Guardar los mejores hiperparámetros en un archivo de texto
with open('model/best_hyperparameters.txt', 'w') as f:
    for key, value in best_params.items():
        f.write(f"{key}: {value}\n")

print("📄 Hiperparámetros guardados en 'model/best_hyperparameters.txt'")
