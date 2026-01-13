# Chest X-Ray Abnormality Detection

This project implements a Deep Learning model (CheXNet / DenseNet121) to detect 14 lung-related abnormalities from chest X-ray images. It includes data preparation, model training, evaluation, and interpretability using Grad-CAM.

## Project Structure

```
chest_xray_detection/
├── src/
│   ├── data/
│   │   ├── dataset.py       # PyTorch Dataset implementation
│   │   ├── path_mapper.py   # Utility to map image paths to CSV
│   ├── modeling/
│   │   ├── chexnet.py       # Model architecture
│   ├── training/
│   │   ├── trainer.py       # Training loop
│   │   ├── evaluate.py      # AUC calculation
│   ├── interpretability/
│   │   ├── gradcam.py       # Grad-CAM visualization
│   ├── main_train.py        # Entry point for training
```

## Setup

1.  **Dependencies**: Ensure PyTorch, TorchVision, Pandas, NumPy, Scikit-learn, and Tqdm are installed.
2.  **Data Preparation**:
    Run the path mapper to index all images:
    ```bash
    python chest_xray_detection/src/data/path_mapper.py
    ```

## Usage

### Training or Evaluation
To train the model (default 5 epochs):
```bash
python chest_xray_detection/src/main_train.py --epochs 5 --batch_size 16
```

### Options
- `--csv_path`: Path to the indexed CSV.
- `--lr`: Learning rate (default 1e-4).
- `--device`: `cuda` or `cpu`.

## Features
- **Robust Data Loading**: Handles recursive image directories and missing files gracefully.
- **Model**: Uses DenseNet121 pretrained on ImageNet, fine-tuned for multi-label classification.
- **Metrics**: Calculates Area Under the ROC Curve (AUC) for each class.
- **Explainability**: Using Grad-CAM to visualize which parts of the X-ray influenced the decision.
