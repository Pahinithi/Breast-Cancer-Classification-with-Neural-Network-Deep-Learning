# Breast Cancer Classification with Neural Network

## Overview

This project implements a Breast Cancer Classification system using a neural network model. It classifies tumors as malignant or benign based on 36 features extracted from breast cancer data. The system is built with Python using libraries such as TensorFlow/Keras for the model, Flask for the web application, and is designed to be user-friendly.

## Features

- **Tumor Classification**: Classifies breast cancer tumors as malignant or benign based on user input.
- **Web Interface**: Provides an interactive form to enter feature values and receive predictions.
- **Neural Network Model**: Utilizes a trained neural network to make predictions.

## Technologies Used

- **Python**: Main programming language.
- **Flask**: Web framework for creating the web application.
- **TensorFlow/Keras**: For building and using the neural network model.
- **Pickle**: To serialize and deserialize the model.
- **HTML/CSS**: For frontend design.

## Setup

### Prerequisites

- Python 3.x
- Flask
- TensorFlow/Keras
- Pickle

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Pahinithi/Breast-Cancer-Classification-with-Neural-Network-Deep-Learning
    cd breast-cancer-classification
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required Python packages:**

    ```bash
    pip install flask tensorflow
    ```

4. **Ensure the model file** (`breast_cancer_classification_model.pkl`) is placed in the `model/` directory.

5. **Run the Flask application:**

    ```bash
    python app.py
    ```

6. **Open a web browser** and go to `http://127.0.0.1:5000` to access the Breast Cancer Classification web interface.

## How It Works

1. **Model Loading**: The pre-trained neural network model is loaded from a `.pkl` file.
2. **Input Collection**: Users input feature values into the web form.
3. **Prediction**: The model processes the input data and makes a prediction.
4. **Result Display**: The prediction result is displayed on the web page.

## Usage

- Enter the values for the following features into the form on the web page:
  - `radius_mean`
  - `texture_mean`
  - `perimeter_mean`
  - `area_mean`
  - `smoothness_mean`
  - `compactness_mean`
  - `concavity_mean`
  - `concave points_mean`
  - `symmetry_mean`
  - `fractal_dimension_mean`
  - `radius_se`
  - `texture_se`
  - `perimeter_se`
  - `area_se`
  - `smoothness_se`
  - `compactness_se`
  - `concavity_se`
  - `concave points_se`
  - `symmetry_se`
  - `fractal_dimension_se`
  - `radius_worst`
  - `texture_worst`
  - `perimeter_worst`
  - `area_worst`
  - `smoothness_worst`
  - `compactness_worst`
  - `concavity_worst`
  - `concave points_worst`
  - `symmetry_worst`
  - `fractal_dimension_worst`
- Click the "Predict" button to receive the classification result.

## Model

The classification model is serialized and saved using Pickle. It includes:

- **Neural Network**: The trained neural network model for classification.
- **Model File**: `breast_cancer_classification_model.pkl`.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. Contributions and suggestions are welcome!

## License

This project is licensed under the MIT License.

## Contact

For any questions or inquiries, please reach out to [nithilan32@gmail.com](mailto:nithilan32@gmail.com).

