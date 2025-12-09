# Nature Scene Classifier

A machine learning project for image classification using FastAI and PyTorch.  
The model classifies natural images into one of the following categories:

- forest  
- beach  
- bird  
- fish  
- other  

This repository contains the full project structure, including the prediction module, web application, CLI interface, and unit tests.  
A deployed version of the model is available on HuggingFace Spaces.

---

## Key Features

- Image classification model built with FastAI and a ResNet18 backbone  
- Clean architecture separating UI, model logic, and utilities  
- Gradio web interface running on HuggingFace Spaces  
- Command-line prediction tool  
- Unit tests with Pytest  
- Reproducible environment using `requirements.txt`  
- Exported FastAI model (`.pkl`) ready for inference  
- CPU-friendly deployment  

---

## Project Structure

```bash
nature-scene-classifier/
│
├── app.py                          # Gradio application
├── cli.py                          # Command-line inference tool
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
│
├── core/                           # Core model logic
│   ├── __init__.py
│   └── predict.py                  # Prediction utilities
│
├── models/                         # Exported FastAI models
│   └── nature_scene_classifier_wsl_v2.pkl
│
├── tests/                          # Unit tests using pytest
│   ├── __init__.py
│   └── test_predict.py
│
└── assets/                         # Sample images and resources
    └── sample.jpg
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/nicoacosta04/nature-scene-classifier.git
cd nature-scene-classifier
```

Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / WSL
.venv\Scripts\activate      # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```
## Running the Gradio Application:
To start the web interface locally:
```bash
python app.py
```

A local server will launch at:
```bash
http://localhost:7860
```

You can upload an image and view the predicted class probabilities.

## Command-Line Interface Usage
The project includes a CLI tool for terminal-based predictions.

Example:
```bash
python cli.py assets/sample.jpg
```

Example output:
```bash
forest: 0.8421
beach: 0.1024
bird: 0.0448
fish: 0.0107
other: 0.0010
```


## Unit testing
Unit tests are included to validate the behavior of the prediction pipeline.

Run the tests using:
```bash
pytest -s
```

Tests verify:
- Correct return type
- Valid probability formats
- Presence of all expected classes
- Prediction pipeline stability

## Model description
The classifier was trained using:
- FastAI (vision learner)
- ResNet18 backbone
- Transfer learning
- Data augmentation
- Automatic image downloading via DuckDuckgo
- Manual verification and cleaning of the dataset
- Data was split into training and validation sets
The final model was exported using:
```bash
learn.export("nature_scene_classifier_wsl_v2.pkl")
```

## Deployment
The web application is deployed on HuggingFace Spaces / https://huggingface.co/spaces/nacostac04/nature-scene-classifier
It uses:
- FastAI for inference
- Gradio blocks for UI rendering
- A lightweight custom prediction pipeline in core/predict.py
Users can upload images and receive classification results directly on the space

## Technologies used
- Python
- FastAI
- Pytorch
- Gradio
- Pytest
- HuggingFace
- DuckDuckGo API

## Skills demonstrated
This project showcases:
- Machine learning model training and evaluation
- Clean modular architecture for ML systems
- Deployment of inference applications
- Testing and validtion of ML pipelines
- Reproducible workflows
- CLI and API desing fundamentals
- Understanding of inference contrainsts and model packing

## Author
Nicolas Acosta
