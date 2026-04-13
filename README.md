# Waste Product Classification Using Transfer Learning (VGG16)

A deep learning project that classifies waste images as **Organic** or **Recyclable** using fine-tuned VGG16 pretrained on ImageNet.

## Overview

Manual waste sorting is labor-intensive and error-prone. This project automates the process using computer vision — given an image of waste, the model predicts whether it belongs in the organic or recyclable bin.

**Transfer learning** is used instead of training from scratch. VGG16 already knows how to detect edges, textures, and shapes from being trained on 1.2M ImageNet images. We unfreeze its last convolutional block (`block5_conv3` onwards) and fine-tune it on waste images, then attach a custom binary classification head.

## Dataset

~1200 labeled waste images split into train/test sets with two classes:
- `O` — Organic (food scraps, leaves, etc.)
- `R` — Recyclable (plastic, metal, paper, etc.)

Downloaded automatically from IBM Skills Network when the notebook is run.

```
o-vs-r-split/
├── train/
│   ├── O/
│   └── R/
└── test/
    ├── O/
    └── R/
```

## Model Architecture

| Layer | Details |
|---|---|
| VGG16 base | Pretrained on ImageNet, blocks 1–4 frozen |
| block5_conv3+ | Unfrozen for domain adaptation |
| Flatten | Converts feature maps to vector |
| Dense(512, ReLU) | Custom classification head |
| Dropout(0.3) | Regularization |
| Dense(1, Sigmoid) | Binary output (O vs R) |

- **Loss:** Binary Crossentropy  
- **Optimizer:** RMSprop (lr=1e-4)  
- **Callbacks:** EarlyStopping, ModelCheckpoint, LearningRateScheduler (exponential decay)

## Requirements

- Python 3.13+
- TensorFlow 2.21.0
- scikit-learn
- matplotlib 3.9.2
- numpy
- tqdm

Install all dependencies by running cell 1 of the notebook.

## Usage

1. Clone the repo
2. Open `Classify_Waste_Products.ipynb` in Jupyter or VS Code
3. Run all cells top to bottom

The notebook will:
1. Install dependencies
2. Download and extract the dataset
3. Build and fine-tune the VGG16 model
4. Plot training loss and accuracy curves
5. Evaluate on the test set (precision, recall, F1)
6. Visualize sample predictions

## Results

The model is evaluated using a full classification report on the held-out test set. Sample predictions are displayed with green titles for correct classifications and red for incorrect ones.

## Project Structure

```
├── Classify_Waste_Products.ipynb   # Main notebook
├── waste_classifier_vgg16.keras    # Saved model (generated after training)
└── README.md
```
