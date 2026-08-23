import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf


from PIL import Image
from streamlit_drawable_canvas import st_canvas


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide"
)


#----------------------------------------------------

def center_by_mass(image):

    moments = cv2.moments(image)

    if moments["m00"] == 0:
        return image

    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]

    shift_x = int(round(14 - center_x))
    shift_y = int(round(14 - center_y))

    matrix = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    centered = cv2.warpAffine(
        image,
        matrix,
        (28, 28)
    )

    return centered



#-----------------------------------------------------


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------


BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "mnist_cnn_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# --------------------------------------------------
# IMAGE PREPROCESSING FUNCTION
# --------------------------------------------------

def preprocess_digit(image):

    # Convert RGBA image to RGB
    rgb_image = image[:, :, :3]

    # Convert RGB to grayscale
    gray = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY
    )

    # Binary threshold
    _, binary = cv2.threshold(
        gray,
        50,
        255,
        cv2.THRESH_BINARY
    )

    # Find coordinates of non-zero pixels
    coords = cv2.findNonZero(binary)

    if coords is None:
        return None

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(coords)

    # Crop digit
    cropped = binary[
        y:y+h,
        x:x+w
    ]

    # ------------------------------------------------
    # MAKE IMAGE SQUARE
    # ------------------------------------------------

    height, width = cropped.shape

    size = max(height, width)

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    # Calculate offsets
    y_offset = (size - height) // 2
    x_offset = (size - width) // 2

    # Place digit in center
    square[
        y_offset:y_offset+height,
        x_offset:x_offset+width
    ] = cropped

    # ------------------------------------------------
    # RESIZE DIGIT TO 20 x 20
    # ------------------------------------------------

    resized_digit = cv2.resize(
        square,
        (20, 20),
        interpolation=cv2.INTER_AREA
    )

    # ------------------------------------------------
    # CREATE 28 x 28 MNIST IMAGE
    # ------------------------------------------------

    mnist_image = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # Put 20x20 digit in center
    mnist_image[4:24, 4:24] = resized_digit

    mnist_image = center_by_mass(
        mnist_image
    )

    # Normalize
    normalized = mnist_image.astype(
        "float32"
    ) / 255.0

    # Add batch and channel dimensions
    model_input = normalized.reshape(
        1,
        28,
        28,
        1
    )

    return {
        "gray": gray,
        "binary": binary,
        "cropped": cropped,
        "square": square,
        "resized": resized_digit,
        "mnist": mnist_image,
        "normalized": normalized,
        "model_input": model_input
    }


# --------------------------------------------------
# DISPLAY PIXEL GRID
# --------------------------------------------------

def show_pixel_grid(image):

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.imshow(
        image,
        cmap="gray",
        interpolation="nearest"
    )

    ax.set_xticks(
        np.arange(-0.5, 28, 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, 28, 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="gray",
        linestyle="-",
        linewidth=0.5
    )

    ax.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False
    )

    return fig


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:
    model = load_model()

except Exception:

    model = None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔢 MNIST Digit Recognition App")

st.markdown(
    """
    Draw a digit between **0 and 9** and see how your drawing
    is transformed step-by-step into a **28 × 28 MNIST image**
    before being sent to the CNN model.
    """
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    stroke_width = st.slider(
        "Stroke Width",
        min_value=5,
        max_value=40,
        value=18
    )

    st.divider()

    st.markdown("### 🧠 Model")

    if model is not None:

        st.success(
            "CNN Model Loaded Successfully"
        )

    else:

        st.error(
            "Model not found!"
        )

        st.info(
            "Run model.py first to train and save the model."
        )


# --------------------------------------------------
# DRAWING CANVAS
# --------------------------------------------------

st.subheader("✍️ Step 1: Draw a Digit")

col1, col2 = st.columns([1, 1])

with col1:

    canvas_result = st_canvas(

        fill_color="rgba(255, 255, 255, 0)",

        stroke_width=stroke_width,

        stroke_color="#FFFFFF",

        background_color="#000000",

        height=280,

        width=280,

        drawing_mode="freedraw",

        key="canvas"

    )


with col2:

    st.info(
        """
        ### How it works

        1. Draw a digit
        2. Convert to grayscale
        3. Detect the digit
        4. Crop empty space
        5. Make the image square
        6. Resize digit to 20 × 20
        7. Add padding
        8. Create 28 × 28 MNIST image
        9. Normalize pixels
        10. Predict the digit
        """
    )


# --------------------------------------------------
# PROCESS DRAWING
# --------------------------------------------------

if canvas_result.image_data is not None:

    image_data = canvas_result.image_data

    # Check whether something is drawn
    rgb = image_data[:, :, :3]

    if np.max(rgb) > 10:

        result = preprocess_digit(
            image_data
        )

        if result is not None:

            st.divider()

            # ----------------------------------------
            # PREPROCESSING VISUALIZATION
            # ----------------------------------------

            st.header(
                "🔬 Step-by-Step Image Preprocessing"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.subheader(
                    "1️⃣ Grayscale"
                )

                st.image(
                    result["gray"],
                    clamp=True
                )

            with c2:

                st.subheader(
                    "2️⃣ Binary Image"
                )

                st.image(
                    result["binary"],
                    clamp=True
                )

            with c3:

                st.subheader(
                    "3️⃣ Cropped Digit"
                )

                st.image(
                    result["cropped"],
                    clamp=True
                )


            c4, c5, c6 = st.columns(3)

            with c4:

                st.subheader(
                    "4️⃣ Square Canvas"
                )

                st.image(
                    result["square"],
                    clamp=True
                )

            with c5:

                st.subheader(
                    "5️⃣ Reduced to 20 × 20"
                )

                st.image(
                    result["resized"],
                    clamp=True
                )

            with c6:

                st.subheader(
                    "6️⃣ MNIST Size: 28 × 28"
                )

                st.image(
                    result["mnist"],
                    clamp=True
                )


            # ----------------------------------------
            # PIXEL GRID
            # ----------------------------------------

            st.divider()

            st.header(
                "🔍 Final 28 × 28 Pixel Visualization"
            )

            st.markdown(
                """
                Each square below represents one pixel.
                This is the actual image structure that is
                sent to the neural network.
                """
            )

            fig = show_pixel_grid(
                result["mnist"]
            )

            st.pyplot(fig)


            # ----------------------------------------
            # IMAGE INFORMATION
            # ----------------------------------------

            st.divider()

            st.header(
                "📊 Image Transformation Summary"
            )

            info_col1, info_col2, info_col3, info_col4 = st.columns(4)

            with info_col1:

                st.metric(
                    "Original Canvas",
                    "280 × 280"
                )

            with info_col2:

                st.metric(
                    "Digit Size",
                    f"{result['cropped'].shape[1]} × "
                    f"{result['cropped'].shape[0]}"
                )

            with info_col3:

                st.metric(
                    "Resized Digit",
                    "20 × 20"
                )

            with info_col4:

                st.metric(
                    "Final MNIST",
                    "28 × 28"
                )


            # ----------------------------------------
            # PREDICTION
            # ----------------------------------------

            st.divider()

            st.header(
                "🤖 Digit Identification"
            )

            if model is not None:

                prediction = model.predict(
                    result["model_input"],
                    verbose=0
                )

                predicted_digit = np.argmax(
                    prediction
                )

                confidence = np.max(
                    prediction
                ) * 100


                pred_col1, pred_col2 = st.columns([1, 2])

                with pred_col1:

                    st.metric(
                        "Predicted Digit",
                        str(predicted_digit)
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )


                with pred_col2:

                    st.subheader(
                        "Prediction Probabilities"
                    )

                    probabilities = prediction[0]

                    chart_data = {
                        str(i): float(probabilities[i])
                        for i in range(10)
                    }

                    st.bar_chart(
                        chart_data
                    )


                # ------------------------------------
                # FINAL DISPLAY
                # ------------------------------------

                st.success(
                    f"""
                    🎉 The CNN model identifies your
                    drawing as digit: {predicted_digit}

                    Confidence: {confidence:.2f}%
                    """
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "MNIST Digit Recognition | Streamlit + OpenCV + TensorFlow CNN"
)