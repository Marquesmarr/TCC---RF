import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mediapipe as mp
from sklearn.metrics import classification_report, f1_score


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

IMG_SIZE = 128
MODEL_PATH = "modelo_binario.keras"
DATA_DIR_MODELO = r"C:\\Users\SS00001419\\Documents\\IA\\RF\\dados\\dados\\modelo"
DATA_DIR_OUTROS = r"C:\\Users\SS00001419\\Documents\\IA\\RF\\dados\\dados\\outros"
CONFIDENCE_THRESHOLD = 0.7

mp_face_detection = mp.solutions.face_detection


def carregar_dados_para_classificador():
    X, y = [], []
    
    for img_name in os.listdir(DATA_DIR_MODELO):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(DATA_DIR_MODELO, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(1)
    
    for img_name in os.listdir(DATA_DIR_OUTROS):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(DATA_DIR_OUTROS, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(0)
    
    return np.array(X, dtype=np.float32) / 255.0, np.array(y)

def criar_modelo_transfer_learning():
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = True
    for layer in base_model.layers[:-20]:  
        layer.trainable = False  

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),  
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),  
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def decidir_treinamento():
    if os.path.exists(MODEL_PATH):
        while True:
            resp = input("Carregar o modelo existente? (s/n): ").strip().lower()
            if resp in ['s']:
                return False
            elif resp in ['n']:
                return True
            else:
                print("Resposta inválida. Digite 's' ou 'n'.")
    else:
        print("Nenhum modelo salvo encontrado. Treinando novo modelo...")
        return True

def processar_frame(frame, face_detection, model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
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
            
            if w > 0 and h > 0:
                rosto = frame[y:y+h, x:x+w]
                rosto_rgb = cv2.cvtColor(rosto, cv2.COLOR_BGR2RGB)
                rosto_resized = cv2.resize(rosto_rgb, (IMG_SIZE, IMG_SIZE))
                rosto_input = np.expand_dims(rosto_resized.astype(np.float32) / 255.0, axis=0)
                
                pred = model.predict(rosto_input, verbose=0)[0][0]
                
                if pred > CONFIDENCE_THRESHOLD:
                    label = "Pessoa Modelo"
                    conf = pred
                    cor = (0, 255, 0)
                else:
                    label = "Outra Pessoa"
                    conf = 1 - pred
                    cor = (0, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    
    return frame



if __name__ == "__main__":
    treinar_novo = decidir_treinamento()

    if treinar_novo:
        print("Carregando dados...")
        X, y = carregar_dados_para_classificador()
        print(f"Carregadas {len(X)} imagens ({np.sum(y)}).")
        
        if len(X) == 0:
            raise ValueError("Nenhuma imagem encontrada nas pastas 'modelo' e 'outros'.")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.25,
            horizontal_flip=True,
            brightness_range=[0.4, 1.4],
            fill_mode='nearest',
            preprocessing_function=lambda x: np.clip(x + np.random.normal(0, 0.02, x.shape), 0, 1)
        )
        datagen.fit(X_train)
        
        model = criar_modelo_transfer_learning()
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
        ]
        
        print("Iniciando treinamento...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        y_val_pred_proba = model.predict(X_val, verbose=0)
        best_threshold = 0.5
        best_f1 = 0
        best_y_pred = None
        
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_val_pred_proba > thresh).astype(int)
            f1 = f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                best_y_pred = y_pred

        print(f"Melhor limiar: {best_threshold:.2f} (F1: {best_f1:.2f})")
        CONFIDENCE_THRESHOLD = best_threshold

        print(classification_report(y_val, best_y_pred, target_names=["Outros", "Modelo"]))
        
        
        print(f"\nModelo salvo em: {MODEL_PATH}")
    else:
        print("Carregando modelo salvo...")
        model = tf.keras.models.load_model(MODEL_PATH)

    print("\nIniciando reconhecimento facial...")
    print("'s' sair.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Não foi possível acessar a webcam.")

    with mp_face_detection.FaceDetection(
        model_selection=1,           
        min_detection_confidence=0.3 
    ) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = processar_frame(frame, face_detection, model)
            cv2.putText(frame, "Reconhecimento Facial Ativo", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Reconhecimento Facial', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Reconhecimento encerrado.")
