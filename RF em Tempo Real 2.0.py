import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mediapipe as mp
import json
from sklearn.metrics import classification_report, f1_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

IMG_SIZE = 128
MODEL_PATH = "modelo_multiclasse.keras"
BASE_DATA_DIR = r"D:\\Nova pasta (3)\\A (TCC) Rede Neural\\RN\\Teste Reconhecimento\\dados" 
LABEL_MAP_PATH = "label_to_name.json"

mp_face_detection = mp.solutions.face_detection

def carregar_dados_multiclasse():
    X, y, label_to_name = [], [], {}
    
    if not os.path.exists(BASE_DATA_DIR):
        raise FileNotFoundError(f"Pasta de dados não encontrada: {BASE_DATA_DIR}")
    
    pastas_pessoas = [d for d in os.listdir(BASE_DATA_DIR) 
                     if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]
    
    if not pastas_pessoas:
        raise ValueError(f"Nenhuma pasta de pessoa encontrada em: {BASE_DATA_DIR}")
    
    label = 0
    for pasta in pastas_pessoas:
        caminho_pasta = os.path.join(BASE_DATA_DIR, pasta)
        label_to_name[label] = pasta
        print(f"Carregando '{pasta}' (rótulo: {label})...")
        
        imagens_pessoa = 0
        for img_name in os.listdir(caminho_pasta):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(caminho_pasta, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(label)
                    imagens_pessoa += 1
        
        print(f"  → {imagens_pessoa} imagens carregadas")
        label += 1
    
    if len(X) == 0:
        raise ValueError("Nenhuma imagem válida encontrada")
    
    return (np.array(X, dtype=np.float32) / 255.0, 
            np.array(y), 
            label_to_name)

def criar_modelo_multiclasse(num_classes):
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
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def decidir_treinamento():
    if os.path.exists(MODEL_PATH):
        while True:
            resp = input("Carregar modelo? (s/n): ").strip().lower()
            if resp in ['s']:
                return False
            elif resp in ['n']:
                return True
    else:
        print("Treinando novo modelo...")
        return True

def processar_frame(frame, face_detection, model, label_to_name, confidence_threshold=0.6):
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
                
                pred_probs = model.predict(rosto_input, verbose=0)[0]
                pred_label = int(np.argmax(pred_probs))
                confidence = float(pred_probs[pred_label])
                
                nome_pessoa = label_to_name.get(pred_label, "Desconhecido")
                
                if confidence > confidence_threshold:
                    cor = (0, 255, 0)  
                    texto = f"{nome_pessoa} ({confidence:.2f})"
                else:
                    cor = (0, 0, 255)  
                    texto = f"Desconhecido ({confidence:.2f})"
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
                cv2.putText(frame, texto, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    
    return frame

if __name__ == "__main__":
    treinar_novo = decidir_treinamento()
    
    if treinar_novo:
        X, y, label_to_name = carregar_dados_multiclasse()
        num_classes = len(label_to_name)
        
        print(f"\nTotal: {len(X)} imagens de {num_classes} pessoas")
        print("Map Nome:")
        for label, name in label_to_name.items():
            count = np.sum(y == label)
            print(f"  {label} -> {name} ({count} imagens)")
        
        with open(LABEL_MAP_PATH, "w", encoding='utf-8') as f:
            json.dump(label_to_name, f, ensure_ascii=False, indent=2)
        print(f"\nMap salvo em: {LABEL_MAP_PATH}")
        
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
            fill_mode='nearest',
            preprocessing_function=lambda x: np.clip(x + np.random.normal(0, 0.02, x.shape), 0, 1)
        )
        datagen.fit(X_train)
        
        model = criar_modelo_multiclasse(num_classes)
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
        ]
        
        print("\nIniciando treinamento...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        nomes_classes = [label_to_name[i] for i in range(num_classes)]
        print("\n" + "="*50)
        print("="*50)
        print(classification_report(y_val, y_val_pred, target_names=nomes_classes))
        
        print(f"\n Modelo salvo em: {MODEL_PATH}")
    
    else:
        print("Carregando...")
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABEL_MAP_PATH, "r", encoding='utf-8') as f:
            label_to_name = json.load(f)
        label_to_name = {int(k): v for k, v in label_to_name.items()}
    
    print("\nIniciando reconhecimento facial...")
    print(" 's'  sair.")
    
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
            
            frame = processar_frame(frame, face_detection, model, label_to_name)
            
            cv2.putText(frame, "Reconhecimento Facial", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Reconhecimento Facial', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n Reconhecimento encerrado.")
