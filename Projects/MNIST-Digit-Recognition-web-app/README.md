# 🔢 MNIST Digit Recognition App

An interactive **Streamlit web application** that allows users to draw handwritten digits and see how the drawing is transformed step by step into an **MNIST-style 28 × 28 image** before being identified by a **Convolutional Neural Network (CNN)**.

The project is designed not only to predict a digit, but also to visually demonstrate the complete preprocessing pipeline used to prepare a hand-drawn image for an MNIST CNN model.

---

## ✨ Features

- ✍️ Draw a digit from **0 to 9** directly in the browser
- 🩶 Convert the drawing to grayscale
- ⚫ Create a binary image using thresholding
- ✂️ Automatically crop empty space around the digit
- ⬜ Convert the cropped digit into a square image
- 📐 Resize the digit to **20 × 20**
- 🔲 Place the digit inside a **28 × 28 MNIST canvas**
- 🎯 Center the digit using image moments / center of mass
- 🔢 Visualize the final **28 × 28 pixel grid**
- 📊 Display image transformation information
- 🤖 Identify the digit using a trained CNN model
- 📈 Display prediction probabilities for digits **0–9**
- 🎨 Includes a colorful and interactive preprocessing interface
- ⚙️ Adjustable drawing stroke width

---

# 🖥️ Application Workflow

```text
✍️ Draw Digit
      ↓
🩶 Convert to Grayscale
      ↓
⚫ Apply Binary Threshold
      ↓
✂️ Crop Empty Space
      ↓
⬜ Make Image Square
      ↓
📐 Resize to 20 × 20
      ↓
🔲 Place on 28 × 28 Canvas
      ↓
🎯 Center by Mass
      ↓
🔢 Normalize Pixel Values
      ↓
🤖 CNN Prediction
```

---

# 📸 Preprocessing Steps

The application displays the image transformation visually.

## 1️⃣ Grayscale

The RGB drawing is converted into a grayscale image.

```python
gray = cv2.cvtColor(
    rgb_image,
    cv2.COLOR_RGB2GRAY
)
```

---

## 2️⃣ Binary Image

A threshold is applied to separate the digit from the black background.

```python
_, binary = cv2.threshold(
    gray,
    50,
    255,
    cv2.THRESH_BINARY
)
```

---

## 3️⃣ Cropped Digit

The application finds all non-zero pixels and removes unnecessary empty space around the digit.

```python
coords = cv2.findNonZero(binary)

x, y, w, h = cv2.boundingRect(coords)

cropped = binary[
    y:y+h,
    x:x+w
]
```

---

## 4️⃣ Square Canvas

The cropped digit may not have equal width and height. Therefore, it is placed inside a square canvas.

```python
height, width = cropped.shape

size = max(height, width)

square = np.zeros(
    (size, size),
    dtype=np.uint8
)
```

The digit is then placed in the center of the square.

---

## 5️⃣ Resize to 20 × 20

The square digit is resized to **20 × 20 pixels**.

```python
resized_digit = cv2.resize(
    square,
    (20, 20),
    interpolation=cv2.INTER_AREA
)
```

---

## 6️⃣ Create 28 × 28 MNIST Image

A blank **28 × 28** image is created.

```python
mnist_image = np.zeros(
    (28, 28),
    dtype=np.uint8
)
```

The 20 × 20 digit is placed inside it with padding.

```python
mnist_image[4:24, 4:24] = resized_digit
```

---

## 🎯 Center by Mass

The application calculates the center of mass of the digit using image moments and shifts the digit toward the center of the 28 × 28 image.

```python
moments = cv2.moments(image)

center_x = moments["m10"] / moments["m00"]
center_y = moments["m01"] / moments["m00"]

shift_x = int(round(14 - center_x))
shift_y = int(round(14 - center_y))
```

The image is shifted using:

```python
cv2.warpAffine()
```

This helps make the user-drawn digit more consistent with the positioning expected by the MNIST CNN model.

---

# 🧠 CNN Model Architecture

The model is trained using the MNIST handwritten digit dataset.

```text
Input Image
28 × 28 × 1
      │
      ▼
Conv2D
32 Filters, 3 × 3
ReLU
      │
      ▼
MaxPooling2D
2 × 2
      │
      ▼
Conv2D
64 Filters, 3 × 3
ReLU
      │
      ▼
MaxPooling2D
2 × 2
      │
      ▼
Flatten
      │
      ▼
Dense
128 Neurons, ReLU
      │
      ▼
Dropout
0.3
      │
      ▼
Dense
10 Neurons, Softmax
      │
      ▼
Predicted Digit
0 – 9
```

The model architecture is:

```python
model = Sequential([

    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(128, activation="relu"),

    Dropout(0.3),

    Dense(10, activation="softmax")
])
```

---

# 📂 Project Structure

```text
MNIST-Digit-Recognition-web-app/
│
├── app.py
│   └── Streamlit web application
│
├── model.py
│   └── CNN model training script
│
├── mnist_cnn_model.keras
│   └── Trained CNN model
│
├── requirements.txt
│   └── Project dependencies
│
└── README.md
    └── Project documentation
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/neerajgaur19/neerajgaur19.git
```

Move to the project directory:

```bash
cd neerajgaur19/Projects/MNIST-Digit-Recognition-web-app
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses:

```text
streamlit==1.62.0
streamlit-drawable-canvas==0.9.3
tensorflow==2.20.0
numpy
opencv-python-headless
Pillow
matplotlib
```

> **Important:** TensorFlow compatibility depends on the Python version and platform. Use a Python version supported by the TensorFlow version specified in `requirements.txt`.

---

# 🏋️ Train the CNN Model

Before running the Streamlit application for the first time, train the CNN model.

Run:

```bash
python model.py
```

The training script will:

1. Download/load the MNIST dataset
2. Normalize pixel values from 0–255 to 0–1
3. Reshape images to `28 × 28 × 1`
4. Build the CNN model
5. Train for 5 epochs
6. Evaluate the model
7. Save the trained model

The trained model is saved as:

```text
mnist_cnn_model.keras
```

The application expects this model file to be located in the same directory as `app.py`.

---

# 🚀 Run the Streamlit Application

After the model has been trained:

```bash
streamlit run app.py
```

Streamlit will start a local server. Open the address shown in the terminal, typically:

```text
http://localhost:8501
```

---

# ☁️ Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Ensure the repository contains:

```text
app.py
model.py
mnist_cnn_model.keras
requirements.txt
```

3. Open Streamlit Community Cloud.
4. Create a new application.
5. Select the GitHub repository.
6. Select the branch.
7. Set the main file path to:

```text
Projects/MNIST-Digit-Recognition-web-app/app.py
```

8. Deploy the application.

---

# 🧮 Model Input Format

The CNN expects input in this shape:

```text
(batch_size, 28, 28, 1)
```

For one image, the application creates:

```python
model_input = normalized.reshape(
    1,
    28,
    28,
    1
)
```

The pixel values are normalized from:

```text
0 – 255
```

to:

```text
0.0 – 1.0
```

---

# 📊 Prediction

The CNN produces probabilities for all ten digits:

```text
Digit 0
Digit 1
Digit 2
Digit 3
Digit 4
Digit 5
Digit 6
Digit 7
Digit 8
Digit 9
```

The digit with the highest probability is selected:

```python
predicted_digit = np.argmax(prediction)
```

The confidence is calculated as:

```python
confidence = np.max(prediction) * 100
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application programming |
| Streamlit | Web application interface |
| streamlit-drawable-canvas | Interactive drawing canvas |
| TensorFlow / Keras | CNN model training and prediction |
| OpenCV | Image preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Pixel-grid visualization |
| Pillow | Image support |

---

# 🎓 Learning Objectives

This project demonstrates several important Machine Learning and Deep Learning concepts:

- Image preprocessing
- Grayscale conversion
- Binary thresholding
- Image cropping
- Image resizing
- Image centering
- Image moments
- Center of mass
- Pixel normalization
- MNIST dataset
- Convolutional Neural Networks
- Conv2D layers
- Max pooling
- Dense layers
- Dropout
- Softmax classification
- Model training
- Model evaluation
- Model deployment using Streamlit

---

# 🔮 Possible Future Improvements

- 🎨 Allow users to choose drawing colors
- 🏆 Show Top-3 predictions
- 📊 Add confidence indicators
- 🎮 Add a digit challenge game
- 🔄 Add a clear/reset button
- 📈 Display model accuracy and training history
- 🧪 Add confusion matrix visualization
- 🔥 Add Grad-CAM or CNN feature-map visualization
- 📱 Improve mobile responsiveness
- 🐳 Add Docker support
- ☁️ Add automated cloud deployment

---

# 👤 Author

**Neeraj Gaur**

---

# ⭐ If You Like This Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
