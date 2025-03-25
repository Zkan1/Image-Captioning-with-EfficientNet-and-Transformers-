import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import clean_caption

def generate_caption(model, tokenizer, photo, max_length):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None or word == '<end>':
            break
        in_text += ' ' + word
    return in_text.replace('<start>', '').strip()

def evaluate_model_bleu(model, mapping, features, tokenizer, max_length, sample_size=100):
    smooth = SmoothingFunction().method1
    total_bleu = 0
    count = 0

    for img_id in list(mapping.keys())[:sample_size]:
        if img_id not in features:
            continue
        photo = features[img_id]
        real_captions = [clean_caption(c).split() for c in mapping[img_id]]
        generated = generate_caption(model, tokenizer, photo, max_length).split()
        bleu_score = sentence_bleu(real_captions, generated, smoothing_function=smooth)
        total_bleu += bleu_score
        count += 1

    print(f"Ortalama BLEU Skoru (@{sample_size} örnek): {total_bleu / count:.4f}")
