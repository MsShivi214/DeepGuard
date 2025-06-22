# DeepGuard

# DeepGuard: Real vs Fake Image Detection

DeepGuard is a web application that uses a deep learning model to classify images as **Real** or **Fake**. It is built with Keras, TensorFlow, and Streamlit.

## Features

- Upload an image and instantly get a prediction: Real or Fake.
- Displays model confidence and raw output for transparency.
- Simple, modern web interface.

## Project Structure

```
DeepGuard/
├── converted_keras (1)/
│   ├── keras_model.h5
│   └── labels.txt
├── Data/
│   ├── Fake/
│   └── Real/
├── deepguard_model.h5
├── image_classifier.py
├── requirements.txt
├── simple_image_test.py
├── streamlit_app.py
└── test_specific_images.py
```

## Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone <repo-url>
   cd DeepGuard
   ```

2. **Install dependencies**  
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**  
   ```bash
   streamlit run streamlit_app.py
   ```
   The app will be available at [http://localhost:8501](http://localhost:8501) (or another port if in use).

## Usage

- Open the app in your browser.
- Upload a `.jpg`, `.jpeg`, or `.png` image.
- View the prediction, confidence, and raw model score.

## Model Details

- The model is a Keras/TensorFlow model saved as `deepguard_model.h5`.
- Class labels are loaded from `converted_keras (1)/labels.txt`.
- Images are resized to 128x128 before prediction.

