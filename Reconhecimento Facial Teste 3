import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import mediapipe as mp

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

IMG_SIZE = 128
BASE_DATA_DIR = r"C:\\Users\\SS00001419\\Documents\\IA\\RF\\dados"
MODELS_DIR = "modelos_pessoais"
os.makedirs(MODELS_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.7
mp_face_detection = mp.solutions.face_detection

def carregar_imagens_pasta(caminho):
    imgs = []
    if not os.path.exists(caminho):
        return np.array(imgs)
    for nome in os.listdir(caminho):
        if nome.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(caminho, nome)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                imgs.append(img)
    return np.array(imgs, dtype=np.float32) / 255.0

def criar_modelo_binario(congelado=True):
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = not congelado

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid', dtype='float32') 
    ])
    return model

def fine_tune_model_progressively(model, X_train, y_train, X_val, y_val, datagen, model_path, epochs=50):
    """
    Treina em 3 fases:
    1. Só o classificador (base congelada)
    2. Descongela últimas 20 camadas
    3. Descongela últimas 40 camadas (opcional, ajustável)
    """
    base_model = model.layers[0]

    base_model.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("Fase 1: Treinando apenas o classificador...")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=max(1, len(X_train) // 32),
        epochs=10,
        validation_data=(X_val, y_val),
        verbose=1
    )

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("Fase 2: Descongelando últimas 20 camadas...")
    callbacks_phase2 = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')
    ]
    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=max(1, len(X_train) // 32),
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=callbacks_phase2,
        verbose=1
    )

    for layer in base_model.layers[:-40]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("Fase 3: Descongelando últimas 40 camadas...")
    callbacks_phase3 = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')
    ]
    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=max(1, len(X_train) // 32),
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=callbacks_phase3,
        verbose=1
    )

def treinar_modelos_por_pessoa():
    pastas_pessoas = [d for d in os.listdir(BASE_DATA_DIR)
                      if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]

    if not pastas_pessoas:
        raise ValueError(f"Nenhuma pasta de pessoa encontrada em: {BASE_DATA_DIR}")

    print(f"Encontradas {len(pastas_pessoas)} pessoas: {pastas_pessoas}")

    for pessoa in pastas_pessoas:
        print(f"\nTreinando modelo para: {pessoa}")
        caminho_pessoa = os.path.join(BASE_DATA_DIR, pessoa)
        X_pos = carregar_imagens_pasta(caminho_pessoa)
        if len(X_pos) == 0:
            print(f" Nenhuma imagem válida para {pessoa}. Pulando.")
            continue

        X_neg_list = []
        for outra in pastas_pessoas:
            if outra != pessoa:
                X_neg_list.append(carregar_imagens_pasta(os.path.join(BASE_DATA_DIR, outra)))

        if not X_neg_list:
            print(f" Nenhuma outra pessoa para compor 'outros'. Pulando {pessoa}.")
            continue

        X_neg = np.concatenate(X_neg_list, axis=0)
        if len(X_neg) == 0:
            print(f" Nenhuma imagem de 'outros' válida. Pulando {pessoa}.")
            continue

        y_pos = np.ones(len(X_pos))
        y_neg = np.zeros(len(X_neg))

        X = np.concatenate([X_pos, X_neg], axis=0)
        y = np.concatenate([y_pos, y_neg], axis=0)

        print(f"  → {len(X_pos)} positivas, {len(X_neg)} negativas")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.25,
            horizontal_flip=True,
            brightness_range=[0.6, 1.4],
            fill_mode='nearest'
        )
        datagen.fit(X_train)

        model = criar_modelo_binario(congelado=True)
        model_path = os.path.join(MODELS_DIR, f"modelo_{pessoa}.keras")

        fine_tune_model_progressively(model, X_train, y_train, X_val, y_val, datagen, model_path)

        print(f"Modelo salvo: {model_path}")

def carregar_todos_modelos():
    modelos = {}
    for arquivo in os.listdir(MODELS_DIR):
        if arquivo.endswith(".keras"):
            nome_pessoa = arquivo.replace("modelo_", "").replace(".keras", "")
            caminho = os.path.join(MODELS_DIR, arquivo)
            try:
                modelos[nome_pessoa] = tf.keras.models.load_model(caminho, compile=False)
                print(f"Carregado modelo para: {nome_pessoa}")
            except Exception as e:
                print(f"Erro ao carregar modelo {arquivo}: {e}")
    return modelos

def preprocessar_rosto_para_modelo(rosto_bgr):
    rosto_rgb = cv2.cvtColor(rosto_bgr, cv2.COLOR_BGR2RGB)
    rosto_resized = cv2.resize(rosto_rgb, (IMG_SIZE, IMG_SIZE))
    rosto_input = np.expand_dims(rosto_resized.astype(np.float32) / 255.0, axis=0)
    return rosto_input

def processar_frame(frame, face_detection, modelos_dict, threshold=0.7, max_faces=2):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if not results.detections:
        return frame

    detections = results.detections[:max_faces]

    for detection in detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)

        if w < 40 or h < 40:
            continue

        x, y = max(0, x), max(0, y)
        w, h = min(w, iw - x), min(h, ih - y)
        if w <= 0 or h <= 0:
            continue

        rosto = frame[y:y+h, x:x+w]
        rosto_input = preprocessar_rosto_para_modelo(rosto)
        melhor_nome = "Desconhecido"
        melhor_conf = 0.0

        for nome, modelo in modelos_dict.items():
            try:
                conf = float(modelo(rosto_input, training=False)[0][0])
                if conf > melhor_conf:
                    melhor_conf = conf
                    melhor_nome = nome
            except Exception as e:
                print(f"Erro ao prever com modelo {nome}: {e}")
                continue

        if melhor_conf > threshold:
            cor = (0, 255, 0)
            texto = f"{melhor_nome} ({melhor_conf:.2f})"
        else:
            cor = (0, 0, 255)
            texto = "Desconhecido"

        cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    return frame

if __name__ == "__main__":
    treinar = False
    if not os.listdir(MODELS_DIR):
        print("Treinando novos modelos...")
        treinar = True
    else:
        resp = input("Treinar novos modelos? (s/n): ").strip().lower()
        treinar = (resp == 's')

    if treinar:
        treinar_modelos_por_pessoa()

    print("\nCarregando modelos...")
    modelos_dict = carregar_todos_modelos()
    if not modelos_dict:
        raise RuntimeError("Nenhum modelo carregado.")

    print(f"\nIniciando reconhecimento facial com {len(modelos_dict)} modelos...")
    print("Pressione 's' para sair.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Não foi possível acessar a webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.3
    ) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Falha ao capturar frame.")
                break

            try:
                frame = processar_frame(frame, face_detection, modelos_dict, CONFIDENCE_THRESHOLD)
            except Exception as e:
                print(f"Erro no processamento do frame: {e}")
                break

            cv2.putText(frame, "Reconhecimento Facial", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Reconhecimento Facial', frame)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nReconhecimento encerrado.")
