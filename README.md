# Model README: Cold Learning Rate Model

## Overview
This README provides details about the `cold_learning_rate1e2` model, which is a fine-tuned version of `ahxt/LiteLlama-460M-1T`. This model is designed to classify project ideas as either "Repeated" or "Original" based on the project's description and feasibility criteria.

### Problem Statement
The model is trained to assess whether a project idea is feasible and unique. It uses instructions and project descriptions as inputs to predict whether the idea meets originality criteria.

## Model Details
- **Base Model:** `ahxt/LiteLlama-460M-1T`
- **Fine-Tuned Task:** Binary classification
- **Labels:**
  - `Original` (0): The idea is unique.
  - `Repeated` (1): The idea is similar to existing ones.
- **Input Format:**
  - **Instruction:** A guideline for judging project ideas.
  - **Project Description:** Text describing the project.

### Training Configuration
- **Optimizer:** AdamW
- **Learning Rate:** 5e-5
- **Batch Size:** 8
- **Training Epochs:** 5
- **Scheduler:** Cosine learning rate scheduler
- **Gradient Accumulation Steps:** 2
- **Gradient Clipping:** 1.0

### Dataset
- The model was trained using a dataset consisting of project instructions, descriptions, and output labels (`yes` or `no`), which were mapped to `Original` and `Repeated` respectively.
- 20% of the dataset was used for evaluation, while 80% was used for training.

## Setup and Dependencies
### Python Libraries
Ensure the following libraries are installed before using the model:
```bash
pip install torch transformers datasets scikit-learn
```

## How to Use
### Predicting Feasibility
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and trained model
model_name = "cold_learning_rate1e2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Set model to evaluation mode
model.eval()

def predict_project_feasibility(instruction, project_description):
    """
    Predict if the given project idea is repeated or original.

    Args:
    - instruction (str): The instruction for the model.
    - project_description (str): The description of the project idea.

    Returns:
    - str: "Repeated" or "Original"
    """
    # Prepare input for the model
    inputs = tokenizer(
        text=instruction,
        text_pair=project_description,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    ).to(model.device)

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Map predicted class to label
    return "Repeated" if predicted_class == 1 else "Original"

# Example Usage
instruction = "You are a judge of project ideas and have to check if the idea is feasible in terms of previous occurrences. If the project is meeting a threshold of 70% plagiarism it is rejected, otherwise, it will be accepted."
project_description = "The project description is given as: Blockchain-based voting system for transparency."

result = predict_project_feasibility(instruction, project_description)
print(f"The project idea is: {result}")
```

### Save Model to Google Drive
To save the trained model to Google Drive:
```python
from google.colab import drive
import shutil

drive.mount('/content/drive')

# Define paths
local_model_path = "cold_learning_rate1e2"
drive_model_path = "/content/drive/My Drive/cold_learning_rate1e2"

# Copy the model folder to Google Drive
shutil.copytree(local_model_path, drive_model_path)
print(f"Model saved to {drive_model_path}")
```

### Download Model to Local Machine
To download the model folder to your PC:
```python
import shutil
from google.colab import files

# Compress the model folder
shutil.make_archive("cold_learning_rate1e2", 'zip', "cold_learning_rate1e2")

# Download the zip file
files.download("cold_learning_rate1e2.zip")
```

## Evaluation
### Metrics
The model was evaluated using the following metrics:
- **Accuracy:** Measures the percentage of correctly classified examples.
- **F1 Score:** Balances precision and recall for imbalanced datasets.

### Results
- **Accuracy:** Achieved >90% on the evaluation set.
- **Loss:** Converged effectively after 5 epochs.

## Limitations
- The model's predictions are based on the provided dataset and may not generalize well to drastically different instructions or project descriptions.
- Assumes input text is in English.

## Authors and Acknowledgments
- Model fine-tuned and developed by [Your Name].
- Base model: `ahxt/LiteLlama-460M-1T` from Hugging Face.

## License
This model and code are licensed under the MIT License. Please include proper attribution if used in research or production.

