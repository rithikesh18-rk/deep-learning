# AI Food Calorie & Nutrition Estimation

An end-to-end Computer Vision and Deep Learning system for classifying Indian food images and estimating standard nutritional values (Calories, Protein, Carbohydrates, Fat) using a fine-tuned EfficientNet-B0 model and a local nutrition lookup database.

---

## 📌 Project Overview

Accurate dietary monitoring is essential for personal health and nutrition tracking. This project provides an intelligent food image classifier trained on popular South Asian dishes. Given an input food photograph, the pipeline:
1. Preprocesses and normalizes the image.
2. Predicts the dish category using a fine-tuned **EfficientNet-B0** convolutional neural network.
3. Computes prediction confidence.
4. Maps the classification result to a local nutrition database to present detailed nutritional metrics per standard serving size.

---

## 🚀 Features

- **High Accuracy Classification**: Fine-tuned EfficientNet-B0 achieving **88.33% validation accuracy**.
- **Instant Nutrition Estimation**: Automated lookup of Calories, Protein, Carbohydrates, and Fat per standard serving.
- **Interactive Flask Web Demo**: Clean, modern web application supporting drag-and-drop file upload with real-time preview.
- **Command Line Interface (CLI)**: Dedicated inference script for terminal-based predictions.
- **Zero Cloud Dependencies**: Operates 100% locally without external APIs, login systems, or database servers.
- **Robust Error Handling**: Automatic validation of file extensions, file sizes, and image integrity (PIL verification).

---

## 🛠️ Technologies Used

- **Deep Learning Framework**: PyTorch, Torchvision
- **Model Architecture**: EfficientNet-B0 (Pretrained on ImageNet)
- **Web Framework**: Flask (Python)
- **Image Processing**: Pillow (PIL), NumPy
- **Frontend / UI**: HTML5, Vanilla CSS, JavaScript (Vanilla)
- **Progress & Utilities**: Tqdm

---

## 🍛 Supported Food Classes

The system currently supports **6 popular food dishes**:

| Food Class | Standard Serving Size | Calories (kcal) | Protein (g) | Carbohydrates (g) | Fat (g) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Biryani** | 1 plate (~300g) | 450 kcal | 18.0 g | 55.0 g | 16.0 g |
| **Chapati** | 1 plate (2 chapatis / ~80g) | 140 kcal | 4.0 g | 24.0 g | 3.0 g |
| **Dosa** | 1 plate (1 medium dosa / ~150g) | 250 kcal | 6.0 g | 45.0 g | 6.0 g |
| **Idli** | 1 plate (2 pieces / ~100g) | 130 kcal | 4.0 g | 28.0 g | 0.5 g |
| **Poori** | 1 plate (2 pooris with bhaji / ~120g) | 300 kcal | 5.0 g | 35.0 g | 16.0 g |
| **Sambar** | 1 bowl (~200ml) | 110 kcal | 5.0 g | 18.0 g | 3.0 g |

---

## 📊 Dataset Information

- **Sources**: Curated combination of Kaggle Indian Food Image Dataset and Food-101 subset.
- **Split**: 80% Training dataset, 20% Validation dataset.
- **Preprocessing**: Resized to 224x224 pixels, normalized using ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).

---

## 🧠 Model Architecture & Training Details

### Backbone Architecture: EfficientNet-B0
EfficientNet uses compound scaling to balance network depth, width, and image resolution efficiently, providing state-of-the-art performance with lower computational cost.

```text
[Input Image (224x224x3)]
         │
         ▼
[EfficientNet-B0 Feature Extractor]
         │
         ▼
[Global Average Pooling & Dropout (0.2)]
         │
         ▼
[Linear Classifier (6 Output Classes)]
```

### Training Strategy & Hyperparameters
- **Transfer Learning**: Backbone pre-trained on ImageNet.
- **Data Augmentation**: Random Horizontal Flip, Random Rotation (±15°), Color Jitter, and Random Resized Crop to improve generalization.
- **Fine-Tuning**: Feature backbone un-frozen and trained with small learning rate (`1e-4`).
- **Optimization**: AdamW optimizer with weight decay (`1e-4`).
- **Early Stopping**: Monitored validation loss to prevent overfitting.
- **Validation Accuracy**: **88.33%** achieved on fine-tuned checkpoint (`models/food_classifier_finetuned.pth`).

---

## 🔄 Project Workflow Diagram

```text
+-----------------------+
|  User Image Upload    |
| (Web UI or CLI Path)  |
+-----------+-----------+
            |
            v
+-----------------------+
| File & Format Check   |  ---> [Reject Invalid / Corrupt File]
|  (PIL Verification)   |
+-----------+-----------+
            | Valid
            v
+-----------------------+
| Preprocess & Normalize|
|  (224x224, Tensor)    |
+-----------+-----------+
            |
            v
+-----------------------+
| EfficientNet-B0 Model |
|  Inference & Softmax  |
+-----------+-----------+
            |
            v
+-----------------------+
| Predicted Class Name  |
| & Confidence Score (%)|
+-----------+-----------+
            |
            v
+-----------------------+
| Local Nutrition       |
| Database Lookup       |
+-----------+-----------+
            |
            v
+-----------------------+
| Final Result Output   |
| (Calories, Protein,   |
|  Carbs, Fat, Serving) |
+-----------------------+
```

---

## 📁 Project Structure

```text
AI-Food-Calorie-Nutrition/
├── app.py                      # Flask web application server
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git exclusion rules
├── src/                        # Core source code
│   ├── config.py               # Hyperparameters & directory paths
│   ├── dataset.py              # PyTorch Dataset & DataLoader loaders
│   ├── model.py                # EfficientNet-B0 model architecture definition
│   ├── nutrition.py            # Local nutrition database & lookup utility
│   ├── predict.py              # Single-image inference pipeline & CLI interface
│   ├── prepare_dataset.py      # Dataset organizing & train/val splitting
│   └── train.py                # Model training loop & checkpoint saver
├── models/                     # Saved model checkpoints
│   ├── food_classifier.pth           # Baseline model checkpoint (~15.6 MB)
│   └── food_classifier_finetuned.pth # Fine-tuned primary checkpoint (~15.6 MB)
├── templates/                  # HTML templates for Flask UI
│   ├── index.html              # Landing & upload page
│   └── result.html             # Classification & nutrition result page
├── static/                     # Web assets & upload directory
│   ├── style.css               # Master application stylesheet
│   └── uploads/                # Directory for user uploaded images
├── scripts/                    # Evaluation & test scripts
│   ├── compare_and_test.py     # Compare baseline vs fine-tuned checkpoint
│   ├── test_nutrition.py       # Unit & integration test for nutrition lookup
│   ├── test_flask_app.py       # Integration test suite for Flask web routes
│   └── test_error_handling.py  # Error handling test suite
└── dataset/                    # Dataset directory structure
    ├── train/                  # Training set split
    └── validation/             # Validation set split
```

---

## 💻 Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/AI-Food-Calorie-Nutrition.git
   cd AI-Food-Calorie-Nutrition
   ```

2. **Create and Activate a Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚡ How to Run Locally

### Option A: Flask Web Demo (Recommended)

1. Start the local Flask web server:
   ```bash
   python app.py
   ```
2. Open your browser and go to:
   ```text
   http://127.0.0.1:5000
   ```
3. Upload any food image (JPG, PNG, WEBP) and click **Analyze Food & Estimate Nutrition**.

### Option B: Command Line Interface (CLI)

Run inference on any individual food image file:
```bash
python src/predict.py --image "dataset/validation/dosa/1000_F_538749842_k3j1R9Q2sWJ9q3s6mS2w0X6t7L.jpg"
```

---

## 🧪 Example Prediction Workflow

```text
=======================================================
 AI Food Calorie & Nutrition Classifier - Result 
=======================================================
 Image Path       : dataset/validation/idli/30_Idly.jpg
 Predicted Food   : Idli
 Confidence Score : 83.47% (0.8347)
-------------------------------------------------------
 NUTRITIONAL INFORMATION (per serving)
 Serving Size     : 1 plate (2 pieces / ~100g)
 Calories         : 130 kcal
 Protein          : 4.0 g
 Carbohydrates    : 28.0 g
 Fat              : 0.5 g
=======================================================
```

---

## ⚠️ Medical & Nutritional Disclaimer

> [!IMPORTANT]
> The nutritional values (Calories, Protein, Carbohydrates, and Fat) displayed by this application are **approximate reference values per standard serving size** retrieved from standard regional nutritional guidelines. They are provided solely for educational and demonstration purposes. Individual nutritional values vary based on ingredients, preparation methods, and portion sizes. This application is **not medical advice** and should not replace professional healthcare or dietary consultation.

---

## 📌 Limitations & Future Improvements

### Current Limitations
- Supports 6 pre-defined South Asian food categories.
- Fixed portion size estimates per serving.
- Model assumes single primary food item per image.

### Future Improvements
- **Multi-Class Detection**: Integrate Object Detection (YOLOv8 / Faster R-CNN) to detect multiple dishes on a single plate.
- **Portion Estimation**: Add depth estimation or reference item scaling for dynamic weight measurement.
- **Expanded Food Database**: Support 50+ regional dishes across global cuisines.
- **Mobile Application**: Port model using PyTorch Mobile / ONNX Runtime for Android & iOS apps.
