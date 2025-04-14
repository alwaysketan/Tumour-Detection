# BrainSight: Brain Tumor Detection using Vision Transformers

![Brain Tumor Classification](https://img.shields.io/badge/domain-medical%20imaging-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)
![Transformers](https://img.shields.io/badge/transformers-4.30+-yellow)

BrainSight is an advanced deep learning system for classifying brain tumors from MRI scans into three categories: meningioma, glioma, and pituitary tumor. This project compares Vision Transformers (ViT) with traditional CNNs and includes an innovative AI-powered radiology report generator.

## Features

- 🧠 **Multi-model comparison**: ViT vs ResNet-50 vs EfficientFormer
- 🖼️ **Data augmentation**: Advanced image transformations for robust training
- 📊 **Performance tracking**: Integrated with Weights & Biases (wandb)
- 📝 **AI report generation**: GPT-4 powered radiology reports
- 🏥 **Clinical relevance**: Designed to assist medical professionals

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alwaysketan/Tumour-Detection.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the [Brain Tumor MRI Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) from Figshare containing 3064 MRI images across three classes:

1. Meningioma (708 images)
2. Glioma (1426 images)
3. Pituitary tumor (930 images)

## Usage

### 1. Data Preparation
```python
# Run the data processing pipeline
python data_preprocessing.py
```

### 2. Model Training
Train Vision Transformer:
```python
python train_vit.py --augmentation True --epochs 10 --batch_size 16
```

Train ResNet-50:
```python
python train_cnn.py --model resnet50 --lr 2e-4
```

### 3. Evaluation
```python
python evaluate.py --model_path ./models/vit_augmented --test_dir ./data/test
```

### 4. Generate Radiology Report
```python
python generate_report.py --image_path sample_mri.jpg --tumor_type glioma
```

## Results

| Model            | Accuracy | Precision | Recall | F1-Score |
|------------------|----------|-----------|--------|----------|
| ViT (no aug)     | 92.3%    | 91.8%     | 92.1%  | 91.9%    |
| ViT (with aug)   | 94.7%    | 94.2%     | 94.5%  | 94.3%    |
| ResNet-50        | 90.1%    | 89.7%     | 90.0%  | 89.8%    |
| EfficientFormer  | 93.5%    | 93.1%     | 93.3%  | 93.2%    |

## Project Structure

```
BrainSight/
├── data/                    # Processed dataset
├── models/                  # Saved model weights
├── notebooks/               # Jupyter notebooks
├── reports/                 # Generated radiology reports
├── src/
│   ├── data_preprocessing.py
│   ├── train_vit.py
│   ├── train_cnn.py
│   ├── evaluate.py
│   └── generate_report.py
├── requirements.txt
└── README.md
```

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- HuggingFace Transformers
- OpenCV
- Weights & Biases (wandb)
- OpenAI API (for report generation)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Contact

For questions or collaborations, please contact:  
[Ketan Dwivedi] - [ketandwivedi05@gmail.com]  
