import re
import pandas as pd
from collections import defaultdict

def clean_caption(text):
    """Caption'ı küçült, noktalama ve fazla boşlukları temizle."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_captions(csv_path, delimiter='|'):
    """CSV dosyasından mapping sözlüğü (image_id → [caption1, caption2...]) oluştur."""
    df = pd.read_csv(csv_path, delimiter=delimiter)
    df.columns = df.columns.str.strip()
    df['comment'] = df['comment'].fillna('')

    mapping = defaultdict(list)
    for _, row in df.iterrows():
        img_id = row['image_name'].strip().split('.')[0]
        caption = row['comment'].strip()
        mapping[img_id].append(f"<start> {caption} <end>")
    return mapping

def clean_mapping(mapping):
    """Mapping sözlüğü içindeki caption'ları temizle (noktalama vs.)."""
    cleaned = {}
    for k in mapping:
        cleaned[k] = [f"<start> {clean_caption(c)} <end>" for c in mapping[k]]
    return cleaned
