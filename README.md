# Emotion-Detection

This project aims to detect emotions from facial images using a Convolutional Neural Network (CNN). The model is trained using the FER-2013 dataset, which consists of 48x48 pixel grayscale images of faces labeled with one of seven emotions. The emotions include:

0: Angry

1: Disgust

2: Fear

3: Happy

4: Sad

5: Surprise

6: Neutral

## Table of Contents
* Project Overview

* Technologies Used

* Dataset

* Setup and Installation

* How to Use

## Project Overview
This project consists of two main components:

* Emotion Detection Model: A CNN model trained to predict emotions from facial expressions based on the FER-2013 dataset.

* FastAPI Backend: An API built with FastAPI to accept image inputs and return emotion predictions.

* Frontend: A simple HTML interface where users can upload images and get predictions.

## Technologies Used
* Python: The main programming language used for the backend and model training.

* TensorFlow/Keras: Used to build and train the CNN model.

* FastAPI: The backend framework for creating the API to handle image uploads and predictions.

* HTML/CSS/Javascript: Used for the simple frontend interface.

* Uvicorn: ASGI server for serving the FastAPI app.

## Dataset
The model is trained using the FER-2013 dataset, which is publicly available and contains labeled images of facial expressions. It consists of:

* Training Set: 28,709 images.

* Test Set: 7,178 images.

The dataset includes images in grayscale format, each of size 48x48 pixels, with faces centered and normalized.

## Setup and Installation
Follow these steps to set up the project on your local machine.

Prerequisites
Make sure you have Python 3.7+ installed.

  1. Clone this repository:
  git clone https://github.com/mariamarouli/Emotion-Detection.git
  cd Emotion-Detection

  2. Create a virtual environment:
  python -m venv .venv

 .venv\Scripts\activate

  4. Install the required dependencies:
  pip install -r requirements.txt

**Make sure to have the trained model file (model.h5) in the project directory. You can train the model or use pre-trained version from this *model.h5* file.**

  4. Running the API
  To start the FastAPI application, run the following command:
  uvicorn main:app --reload
  This will start the server at http://127.0.0.1:8000.


## How to Use
* Upload an Image: Navigate to the frontend (provided in index.html), where you can upload a facial image.

* Prediction: The image will be sent to the FastAPI backend, which will preprocess the image, make a prediction using the trained model, and return the emotion.


