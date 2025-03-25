import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

def build_feature_extractor():
    """EfficientNetB5 tabanlı özellik çıkarım modeli döndürür."""
    base_model = EfficientNetB5(weights="imagenet", include_top=False, input_shape=(456, 456, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def extract_features(model, image_dir, target_size=(456, 456)):
    """Belirtilen klasördeki görsellerden özellik vektörleri çıkarır."""
    features = {}
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in tqdm(image_files, desc="Özellik çıkarılıyor"):
        img_path = os.path.join(image_dir, img_name)
        image = load_img(img_path, target_size=target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature_vector = model.predict(image, verbose=0)
        
        image_id = os.path.splitext(img_name)[0]
        features[image_id] = feature_vector
    return features
