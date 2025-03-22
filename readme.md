#  Image Captioning with EfficientNet and Transformers

A deep learning project that combines vision and language to generate product descriptions from images. This repository explores an image captioning pipeline using EfficientNetB5 as the visual feature extractor and Transformer-based decoders for generating fluent and context-aware captions.

---

##  Project Goals

- Build an end-to-end image captioning system.
- Use **EfficientNetB5** for image feature extraction.
- Compare **LSTM and Transformer decoders**.
- Pretrain on **Flickr8k dataset**, finetune on **OpenFoodFacts**.
- Create a web interface for demo.

---

##  Project Structure

```bash
.
├── data/
│   ├── flickr8k/
│   └── openfoodfacts/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_lstm_caption_model.ipynb
│   ├── 04_transformer_caption_model.ipynb
├── models/
│   └── best_model.h5
├── app/
│   └── gradio_interface.py
├── utils/
│   └── tokenizer_utils.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
```

---

##  Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your_username/Image-Captioning-with-EfficientNet-and-Transformers.git
cd Image-Captioning-with-EfficientNet-and-Transformers
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

##  Datasets

- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [OpenFoodFacts API](https://world.openfoodfacts.org/data)

Download the datasets and place them under the `data/` directory accordingly.

---

##  Models
- **Baseline Model**: EfficientNetB5 + LSTM Decoder
- **Advanced Model**: EfficientNetB5 + Transformer Decoder

Both models are trained using caption sequences tokenized and padded to uniform length.

---



## .gitignore
```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
venv/
.env
*.env

# Jupyter Notebook Checkpoints
.ipynb_checkpoints

# Data & Model Artifacts
/data/
/models/
*.h5
*.pkl
*.npz

# Logs
*.log
logs/

# OS files
.DS_Store
Thumbs.db
```

---


