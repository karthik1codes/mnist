# MNIST Digit Recognition Web App

This is a web application that allows users to draw digits and uses a trained MNIST model to predict the drawn number. The application features a user-friendly interface with a drawing canvas and real-time predictions.

## Features

- Interactive drawing canvas
- Real-time digit recognition
- Confidence score display
- Mobile-friendly interface
- Clear and modern UI

## Setup Instructions

Run on Python 3.10 version

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the Flask application:
```bash
python app.py
```

4. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Draw a digit (0-9) in the canvas using your mouse or touch screen
2. Click the "Predict" button to get the prediction
3. The predicted digit and confidence score will be displayed below the canvas
4. Use the "Clear" button to erase the canvas and draw a new digit

## Technical Details

- The model is a Convolutional Neural Network (CNN) trained on the MNIST dataset
- The web interface uses HTML5 Canvas for drawing
- The backend is built with Flask and TensorFlow
- The application automatically normalizes and resizes the drawn image to match the MNIST format 