seed = 9921378
output_vit_no_aug = '...'
output_vit_no_aug2 = '...'
output_vit_no_aug3 = '...'
output_vit_with_aug = '...'
output_cnn_no_aug = '...'
output_cnn_with_aug = '...'
data_path = '...'
!pip install datasets
!pip install transformers[torch]
!pip install accelerate -U
!pip install wandb

!pip install imgaug
# Load the dataset
import os
import numpy as np

# Create the required directories
os.makedirs(data_path + '/brain_tumor_dataset', exist_ok=True)
os.makedirs(data_path + '/brain_dataset/train', exist_ok=True)
os.makedirs(data_path +'/brain_dataset/test', exist_ok=True)

# Names of tumor types
tumor_types = ['meningioma','glioma','pituitary tumor']

# Create directories for each tumor type in both train and test sets
for tumor_type in tumor_types:
  os.makedirs(data_path + f'/brain_dataset/test/{tumor_type}', exist_ok=True)
  os.makedirs(data_path + f'/brain_dataset/train/{tumor_type}', exist_ok=True)

# Install necessary Python libraries
!pip install hdf5storage


# Change directory
%cd $data_path'/brain_tumor_dataset'

# Download the dataset
!wget https://ndownloader.figshare.com/articles/1512427/versions/5

# Unzip the dataset and delete the zip
!unzip 5 && rm 5

# Concatenate the multiple zipped data in a single zip
!cat brainTumorDataPublic_* > brainTumorDataPublic_temp.zip
!zip -FF brainTumorDataPublic_temp.zip --out data.zip

# Remove the temporary files
!rm brainTumorDataPublic_*

# Unzip the full archive and delete it
!unzip data.zip -d data && rm data.zip

# Check that "data" contains 3064 files
!ls data | wc -l

# Change directory to previous
%cd ..

# Clone the GitHub repository containing the script for converting the MATLAB files to NumPy
!git clone https://github.com/guillaumefrd/brain-tumor-mri-dataset.git
!cp brain-tumor-mri-dataset/matlab_to_numpy.py $data_path

# Run the Python script to convert the MATLAB files to NumPy
!python $data_path'/matlab_to_numpy.py' $data_path'/brain_tumor_dataset'
# Load labels, images, and masks
%cd $data_path'/brain_tumor_dataset'

labels = np.load('labels.npy') - 1  # Subtracting 1 from labels to make them 0-indexed
images = np.load('images.npy')
masks = np.load('masks.npy')
from PIL import Image
# Normalize images and convert to PIL format
def normalize_and_convert_images_to_pil(images):
  normalized_pil_images = []
  for image in images:
    normalized_image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))
    normalized_pil_images.append(Image.fromarray(normalized_image.astype(np.uint8)).convert('RGB'))
  return normalized_pil_images
# Split the dataset into train and test
train_test_split_ratio = 0.8
num_samples = labels.shape[0]
indices = np.random.RandomState(seed).permutation(num_samples)
train_indices = indices[:int(num_samples * train_test_split_ratio)]
test_indices = indices[int(num_samples * train_test_split_ratio):]
# Convert images to PIL format
pil_images = normalize_and_convert_images_to_pil(images)
# Save images to respective directories
index_to_tumor_type = {i: tumor_type for i, tumor_type in enumerate(tumor_types)}
for train_index in train_indices:
  pil_images[train_index].save(data_path + f'/brain_dataset/train/{index_to_tumor_type[labels[train_index]]}/{train_index}.png')
for test_index in test_indices:
  pil_images[test_index].save(data_path + f'/brain_dataset/test/{index_to_tumor_type[labels[test_index]]}/{test_index}.png')
# Imports
# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from datasets import load_dataset, load_metric

from transformers import TrainingArguments, ViTFeatureExtractor, ViTForImageClassification, Trainer

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Lambda
from torchvision.datasets import ImageFolder

from accelerate import Accelerator
import imgaug.augmenters as iaa

import wandb
labels_names = ['meningioma','glioma','pituitary tumor']
# Helper Functions
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

metric = load_metric("accuracy")

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
#confusion matrix functions
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn import metrics as met

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    cm = confusion_matrix(p.label_ids, preds)
    return cm

def print_confusion_matrix(cm, labels_names):
    df_cm = pd.DataFrame(cm, index=labels_names, columns=labels_names)
    print(df_cm)

# ViT

## Transforms
# Define the augmentation pipeline with imgaug
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip with probability of 0.5
    iaa.Sometimes(0.3, iaa.GaussianBlur((0, 1.0))),  # apply Gaussian blur with probability of 0.3
    iaa.Sometimes(0.5, iaa.SaltAndPepper(0.05)),  # Add Salt and Pepper noise with probability of 0.5
    iaa.Sometimes(0.3, iaa.AdditivePoissonNoise(lam=(0, 30))),  # Add Poisson noise with probability of 0.3
])

# imgaug works with numpy images (HWC) but torchvision with PIL images, so we have to convert
imgaug_transform = Lambda(lambda img: augmenter.augment_image(np.array(img)))
# Define transformations for the train dataset (without augmentation)
train_transforms_without_aug = transforms.Compose([
    transforms.Resize((224, 224),interpolation=Image.BILINEAR),  # Resize 512x512 images to 224x224 using bilinear interpolation
    transforms.ToTensor(),  # Convert image to PyTorch tensor and automatically rescale to [0,1] by dividing by 255
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensor to range [-1,1]
])

# Define transformations for the train dataset (including augmentation)
train_transforms_with_aug = transforms.Compose([
    transforms.Resize((224, 224),interpolation=Image.BILINEAR),  # Resize 512x512 images to 224x224 using bilinear interpolation
    imgaug_transform,
    transforms.ToTensor(),  # Convert image to PyTorch tensor and automatically rescale to [0,1] by dividing by 255
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensor to range [-1,1]
])

# Define transformations for the test dataset
test_transforms = transforms.Compose([
    transforms.Resize((224, 224),interpolation=Image.BILINEAR),  # Resize 512x512 images to 224x224 using bilinear interpolation
    transforms.ToTensor(),  # Convert image to PyTorch tensor and automatically rescale to [0,1] by dividing by 255
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensor to range [-1,1]
])
## Without augmentation
ViT_base = 'google/vit-base-patch16-224-in21k'

# Load the datasets from folders and process the dataset with the transform
train_dataset = ImageFolder(data_path + '/brain_dataset/train', transform=train_transforms_without_aug)
test_dataset = ImageFolder(data_path + '/brain_dataset/test', transform=test_transforms)
# define the model
def model_init():
  return ViTForImageClassification.from_pretrained(
    ViT_base,
    num_labels=len(labels_names),
    id2label={str(i): tumor_type for i, tumor_type in enumerate(labels_names)},
    label2id={tumor_type: i for i, tumor_type in enumerate(labels_names)},
)

model = model_init()

# Initialize a new run
wandb.init(project="vit_no_aug")

# Define the TrainingArguments
training_args = TrainingArguments(
  output_dir=output_vit_no_aug,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=5,
  fp16=True,
  save_steps=100,#100
  eval_steps=50,#100
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='wandb',
  load_best_model_at_end=True,
  seed = seed
)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator= lambda samples: {'pixel_values': torch.stack([sample[0] for sample in samples]),
                                   'labels': torch.tensor([sample[1] for sample in samples])},
)

#final
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

wandb.finish()
#display Confusion Matrix
pred = trainer.predict(test_dataset)

cm_display = met.ConfusionMatrixDisplay(confusion_matrix = compute_metrics(pred), display_labels = labels_names)

cm_display.plot()
plt.show()

## With augmentation
ViT_base = 'google/vit-base-patch16-224-in21k'

# Load the datasets from folders and process the dataset with the transform
train_dataset = ImageFolder(data_path + '/brain_dataset/train', transform=train_transforms_with_aug)
test_dataset = ImageFolder(data_path + '/brain_dataset/test', transform=test_transforms)
# define the model
model = ViTForImageClassification.from_pretrained(
    ViT_base,
    num_labels=len(labels_names),
    id2label={str(i): c for i, c in enumerate(labels_names)},
    label2id={c: str(i) for i, c in enumerate(labels_names)}
)

# Initialize a new run
wandb.init(project="vit_with_aug")

# Define the TrainingArguments
training_args = TrainingArguments(
  output_dir=output_vit_with_aug,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=5,
  fp16=True,
  save_steps=100,#100
  eval_steps=50,#100
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='wandb',
  load_best_model_at_end=True,
  seed = seed
)


trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator= lambda samples: {'pixel_values': torch.stack([sample[0] for sample in samples]),
                                   'labels': torch.tensor([sample[1] for sample in samples])},
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

wandb.finish()
#display Confusion Matrix
pred = trainer.predict(test_dataset)

cm_display = met.ConfusionMatrixDisplay(confusion_matrix = compute_metrics(pred), display_labels = labels_names)

cm_display.plot()
plt.show()

## Load pretrain

from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained('/content/drive/MyDrive/colab_files/brain_tumor_dataset/vit-model')
# CNN

## Transforms
torch.cuda.empty_cache()
# Define the augmentation pipeline with imgaug
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip with probability of 0.5
    iaa.Sometimes(0.3, iaa.GaussianBlur((0, 1.0))),  # apply Gaussian blur with probability of 0.3
    iaa.Sometimes(0.5, iaa.SaltAndPepper(0.05)),  # Add Salt and Pepper noise with probability of 0.5
    iaa.Sometimes(0.3, iaa.AdditivePoissonNoise(lam=(0, 30))),  # Add Poisson noise with probability of 0.3
])

# imgaug works with numpy images (HWC) but torchvision with PIL images, so we have to convert
imgaug_transform = Lambda(lambda img: augmenter.augment_image(np.array(img)))
# Define transformations for the train and test datasets
# Define the cropping percentage
crop_percentage = 0.875

# Define transformations for the train dataset
train_transforms_without_aug = transforms.Compose([
    transforms.CenterCrop((int(512 * crop_percentage),int(512 * crop_percentage))),
    transforms.Resize((224, 224),interpolation=Image.BICUBIC), # Resize 508x508 images to 224x224 using BICUBIC interpolation
    transforms.ToTensor(), # Convert image to PyTorch tensor and automatically rescale to [0,1] by dividing by 255
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define transformations for the train dataset (including augmentation)
train_transforms_with_aug = transforms.Compose([
    transforms.CenterCrop((int(512 * crop_percentage),int(512 * crop_percentage))),
    transforms.Resize((224, 224),interpolation=Image.BICUBIC), # Resize 508x508 images to 224x224 using BICUBIC interpolation
    imgaug_transform,
    transforms.ToTensor(), # Convert image to PyTorch tensor and automatically rescale to [0,1] by dividing by 255
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Define transformations for the test dataset

test_transforms_without_aug = transforms.Compose([
    transforms.CenterCrop((int(512 * crop_percentage),int(512 * crop_percentage))),
    transforms.Resize((224, 224),interpolation=Image.BICUBIC), # Resize 508x508 images to 224x224 using BICUBIC interpolation
    transforms.ToTensor(), # Convert image to PyTorch tensor and automatically rescale to [0,1] by dividing by 255
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

## Without augmentation
CNN_base = 'microsoft/resnet-50'

# Load the datasets from folders and process the dataset with the transform
train_dataset = ImageFolder(data_path + '/brain_dataset/train', transform=train_transforms_without_aug)
test_dataset = ImageFolder(data_path + '/brain_dataset/test', transform=test_transforms_without_aug)
# define the model
def model_init():
  return ViTForImageClassification.from_pretrained(
    CNN_base,
    num_labels=len(labels_names),
    id2label={str(i): tumor_type for i, tumor_type in enumerate(labels_names)},
    label2id={tumor_type: i for i, tumor_type in enumerate(labels_names)},
)

model = model_init()
# Initialize a new run
wandb.init(project="cnn_no_aug")

# Define the TrainingArguments
training_args = TrainingArguments(
  output_dir=output_cnn_no_aug,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=5, #4
  fp16=True,
  save_steps=100,#100
  eval_steps=50,#100
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='wandb',
  load_best_model_at_end=True,
  seed = seed
)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator= lambda samples: {'pixel_values': torch.stack([sample[0] for sample in samples]),
                                   'labels': torch.tensor([sample[1] for sample in samples])},
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

wandb.finish()
#display Confusion Matrix
pred = trainer.predict(test_dataset)

cm_display = met.ConfusionMatrixDisplay(confusion_matrix = compute_metrics(pred), display_labels = labels_names)

cm_display.plot()
plt.show()

## With augmentation
CNN_base = 'microsoft/resnet-50'

# Load the datasets from folders and process the dataset with the transform
train_dataset = ImageFolder(data_path + '/brain_dataset/train', transform=train_transforms_with_aug)
test_dataset = ImageFolder(data_path + '/brain_dataset/test', transform=test_transforms_without_aug)
# define the model
def model_init():
  return ViTForImageClassification.from_pretrained(
    CNN_base,
    num_labels=len(labels_names),
    id2label={str(i): tumor_type for i, tumor_type in enumerate(labels_names)},
    label2id={tumor_type: i for i, tumor_type in enumerate(labels_names)},
)

model = model_init()
# Initialize a new run
wandb.init(project="cnn_with_aug")

# Define the TrainingArguments
training_args = TrainingArguments(
  output_dir=output_cnn_with_aug,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=5, #4
  fp16=True,
  save_steps=100,#100
  eval_steps=50,#100
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='wandb',
  load_best_model_at_end=True,
  seed = seed
)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator= lambda samples: {'pixel_values': torch.stack([sample[0] for sample in samples]),
                                   'labels': torch.tensor([sample[1] for sample in samples])},
)


train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

wandb.finish()
#display Confusion Matrix
pred = trainer.predict(test_dataset)

cm_display = met.ConfusionMatrixDisplay(confusion_matrix = compute_metrics(pred), display_labels = labels_names)

cm_display.plot()
plt.show()
#EfficientFormer
###Import and create model
from transformers import EfficientFormerForImageClassification
from transformers import AutoImageProcessor

# define the EfficientFormer Model
def model_init():
  model = EfficientFormerForImageClassification.from_pretrained("snap-research/efficientformer-l1-300")
  model.config.label2id = {tumor_type: i for i, tumor_type in enumerate(labels_names)}
  model.config.num_labels = len(labels_names)
  model.config.id2label = {str(i): tumor_type for i, tumor_type in enumerate(labels_names)}
  return model
##Create Transforms
torch.cuda.empty_cache()
# Define the augmentation pipeline with imgaug
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip with probability of 0.5
    iaa.Sometimes(0.3, iaa.GaussianBlur((0, 0.2))),  # apply Gaussian blur with probability of 0.3
    iaa.Sometimes(0.5, iaa.SaltAndPepper(0.05)),  # Add Salt and Pepper noise with probability of 0.5
    iaa.Sometimes(0.1, iaa.AdditivePoissonNoise(lam=(0, 30))),  # Add Poisson noise with probability of 0.3
])


# imgaug works with numpy images (HWC) but torchvision with PIL images, so we have to convert
imgaug_transform = Lambda(lambda img: augmenter.augment_image(np.array(img)))

# Define transformations for the train dataset (without augmentation)
train_transforms_without_aug = transforms.Compose([
    transforms.Resize((224, 224),interpolation=Image.BILINEAR),  # Resize 512x512 images to 224x224 using bilinear interpolation
    transforms.ToTensor(),  # Convert image to PyTorch tensor and automatically rescale to [0,1] by dividing by 255
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensor to range [-1,1]
])

# Define transformations for the train dataset (including augmentation)
train_transforms_with_aug = transforms.Compose([
    transforms.Resize((224, 224),interpolation=Image.BILINEAR),  # Resize 512x512 images to 224x224 using bilinear interpolation
    imgaug_transform,
    transforms.ToTensor(),  # Convert image to PyTorch tensor and automatically rescale to [0,1] by dividing by 255
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensor to range [-1,1]
])


# Define transformations for the test dataset
test_transforms = transforms.Compose([
    transforms.Resize((224, 224),interpolation=Image.BILINEAR),  # Resize 512x512 images to 224x224 using bilinear interpolation
    transforms.ToTensor(),  # Convert image to PyTorch tensor and automatically rescale to [0,1] by dividing by 255
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensor to range [-1,1]
])
## Without Augmentation
#This needs to be re-defined when the notebook is this long:
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

metric = load_metric("accuracy")

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
# Load the datasets from folders and process the dataset with the transform
train_dataset = ImageFolder(data_path + '/brain_dataset/train', transform=train_transforms_without_aug)
test_dataset = ImageFolder(data_path + '/brain_dataset/test', transform=test_transforms)
# Initialize a new run

# Define the TrainingArguments
training_args = TrainingArguments(
  output_dir=output_vit_no_aug,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=15,
  fp16=True,
  save_steps=100,#100
  eval_steps=50,#100
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='wandb',
  load_best_model_at_end=True,
  seed = seed
)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator= lambda samples: {'pixel_values': torch.stack([sample[0] for sample in samples]),
                                   'labels': torch.tensor([sample[1] for sample in samples])},
)
wandb.init(project="eff_vit_with_aug")
#Train model:
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
wandb.finish()
## With Augmentation
# Load the datasets from folders and process the dataset with the transform
train_dataset = ImageFolder(data_path + '/brain_dataset/train', transform=train_transforms_with_aug)
test_dataset = ImageFolder(data_path + '/brain_dataset/test', transform=test_transforms)
# Initialize a new run

# Define the TrainingArguments
training_args = TrainingArguments(
  output_dir=output_vit_no_aug,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=15,
  fp16=True,
  save_steps=100,#100
  eval_steps=50,#100
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='wandb',
  load_best_model_at_end=True,
  seed = seed
)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator= lambda samples: {'pixel_values': torch.stack([sample[0] for sample in samples]),
                                   'labels': torch.tensor([sample[1] for sample in samples])},
)
wandb.init(project="eff_vit_with_aug")
#Train model:
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
wandb.finish()
wandb.init(project="eff_vit_with_aug")
#Train model:
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
wandb.finish()
!pip install openai
import json
from openai import OpenAI
import os
from google.colab import userdata

MODEL = "gpt-4o"

client = OpenAI(api_key=userdata.get('openai'))
from IPython.display import Image, display
import base64

IMAGE_PATH = "/content/15381tn.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image(IMAGE_PATH)
display(Image(IMAGE_PATH))
tumor_type='glioma'
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a highly knowledgeable Radiology specializing in radiology and oncology. "
            "Your task is to analyze medical findings and produce structured reports for MRI scans of the brain."},
        {"role": "user", "content": [
            {"type": "text", "text": f"""Please analyze the attached MRI brain of brain tumor of type '{tumor_type}'  and generate a structured medical report. "
                "Include the following details: \n"
                "- Study Type\n"
                "- Findings (lesion location, characteristics, mass effect, differential diagnosis)\n"
                "- Impression\n\n"
                "The tumor type is '{tumor_type}'."""},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]}
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)
tumor_type='meningioma'
from IPython.display import Image, display
import base64

IMAGE_PATH = "/content/download (2).jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image(IMAGE_PATH)
display(Image(IMAGE_PATH))
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a highly knowledgeable medical assistant specializing in radiology and oncology. "
            "Your task is to analyze medical findings and produce structured reports for MRI scans of the brain."},
        {"role": "user", "content": [
            {"type": "text", "text": f"""Please analyze the attached MRI brain of brain tumor of type '{tumor_type}'  and generate a structured medical report. "
                "Include the following details: \n"
                "- Study Type\n"
                "- Findings (lesion location, characteristics, mass effect, differential diagnosis)\n"
                "- Impression\n\n"
                "The tumor type is '{tumor_type}'."""},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]}
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)
