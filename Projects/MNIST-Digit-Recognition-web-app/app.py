import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import csv

from streamlit_drawable_canvas import st_canvas


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .stDeployButton { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }

.mode-card {
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}    

.prediction-box {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
}

.big-digit {
    font-size: 100px;
    font-weight: bold;
    margin: 0;
}


/* Center Streamlit images */
div[data-testid="stImage"] {
    display: flex;
    justify-content: center;
}

/* Center image captions */
div[data-testid="stImage"] + div {
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "mnist_cnn_model.keras"

TRAINING_DATA_DIR = BASE_DIR / "training_data"

TRAINING_DATA_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "all_predictions" not in st.session_state:
    st.session_state.all_predictions = {}

if "collection_count" not in st.session_state:
    st.session_state.collection_count = 0

if "training_canvas_reset" not in st.session_state:
    st.session_state.training_canvas_reset = 0

# ============================================================
# CENTER DIGIT BY MASS
# ============================================================

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


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_digit(image):

    # --------------------------------------------------------
    # 1. RGBA → RGB
    # --------------------------------------------------------

    rgb_image = image[:, :, :3]

    # --------------------------------------------------------
    # 2. RGB → GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY
    )

    # --------------------------------------------------------
    # 3. BINARY IMAGE
    # --------------------------------------------------------

    _, binary = cv2.threshold(
        gray,
        50,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # 4. FIND DIGIT
    # --------------------------------------------------------

    coords = cv2.findNonZero(binary)

    if coords is None:
        return None

    # --------------------------------------------------------
    # 5. CROP DIGIT
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(coords)

    cropped = binary[
        y:y+h,
        x:x+w
    ]

    # --------------------------------------------------------
    # 6. MAKE SQUARE
    # --------------------------------------------------------

    height, width = cropped.shape

    size = max(
        height,
        width
    )

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    y_offset = (size - height) // 2
    x_offset = (size - width) // 2

    square[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ] = cropped

    # --------------------------------------------------------
    # 7. RESIZE TO 20 × 20
    # --------------------------------------------------------

    resized_digit = cv2.resize(
        square,
        (20, 20),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # 8. CREATE 28 × 28 MNIST IMAGE
    # --------------------------------------------------------

    mnist_before_center = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    mnist_before_center[
        4:24,
        4:24
    ] = resized_digit

    # --------------------------------------------------------
    # 9. CENTER BY MASS
    # --------------------------------------------------------

    mnist_image = center_by_mass(
        mnist_before_center
    )

    # --------------------------------------------------------
    # 10. NORMALIZE
    # --------------------------------------------------------

    normalized = (
        mnist_image.astype("float32") / 255.0
    )

    # --------------------------------------------------------
    # 11. CNN INPUT
    # --------------------------------------------------------

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

        "before_center": mnist_before_center,

        "mnist": mnist_image,

        "normalized": normalized,

        "model_input": model_input
    }


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


try:

    model = load_model()

except Exception as e:

    model = None

    model_error = str(e)


# ============================================================
# PREDICT DIGIT
# ============================================================

def predict_digit(model, model_input):

    prediction = model.predict(
        model_input,
        verbose=0
    )

    predicted_digit = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction) * 100
    )

    probabilities = prediction[0]

    return (
        predicted_digit,
        confidence,
        probabilities
    )


# ============================================================
# PIXEL GRID
# ============================================================

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


# ============================================================
# DISPLAY PREPROCESSING
# ============================================================

def display_preprocessing(result):

    st.divider()

    st.header(
        "🔬 Step-by-Step Image Preprocessing"
    )

    # --------------------------------------------------------
    # ROW 1
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader(
            "1️⃣ Grayscale"
        )

        st.image(
            result["gray"],
            clamp=True,
            use_container_width=True
        )

    with c2:

        st.subheader(
            "2️⃣ Binary Image"
        )

        st.image(
            result["binary"],
            clamp=True,
            use_container_width=True
        )

    with c3:
        st.subheader(
            "3️⃣ Cropped Digit"
        )

        st.image(
            result["cropped"],
            clamp=True,
            use_container_width=True
        )

    # --------------------------------------------------------
    # ROW 2
    # --------------------------------------------------------

    c4, c5, c6 = st.columns(3)

    with c4:

        st.subheader(
            "4️⃣ Square Image"
        )

        st.image(
            result["square"],
            clamp=True,
            use_container_width=True
        )

    with c5:

        st.subheader(
            "5️⃣ Reduced to 20 × 20"
        )

        st.image(
            result["resized"],
            clamp=True,
            use_container_width=True
        )

    with c6:

        st.subheader(
            "6️⃣ Final MNIST 28 × 28"
        )

        st.image(
            result["mnist"],
            clamp=True,
            use_container_width=True
        )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

def display_prediction(
    predicted_digit,
    confidence,
    probabilities
):

    st.divider()

    st.header(
        "🤖 Digit Identification"
    )

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        st.metric(
            "Predicted Digit",
            str(predicted_digit)
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with col2:

        st.subheader(
            "Prediction Probabilities"
        )

        chart_data = {

            str(i): float(
                probabilities[i]
            )

            for i in range(10)
        }

        st.bar_chart(
            chart_data
        )

    st.markdown(
        f"""
        <div class="prediction-box">
            <h2>🤖 AI Prediction</h2>
            <h1 style="font-size:90px; margin:0;">{predicted_digit}</h1>
            <h3>Confidence: {confidence:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔢 MNIST Digit Recognition"
)

st.markdown(
    """
    Draw handwritten digits and see how a CNN
    processes the image before making a prediction.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Settings"
    )

    stroke_width = st.slider(
        "Stroke Width",
        min_value=5,
        max_value=40,
        value=18
    )

    st.divider()

    st.header(
        "🧠 CNN Model"
    )

    if model is not None:

        st.success(
            "CNN Model Loaded"
        )

    else:

        st.error(
            "CNN Model Not Found"
        )

        st.info(
            "Make sure mnist_cnn_model.keras "
            "exists beside app.py."
        )


# ============================================================
# MODE SELECTION
# ============================================================

st.subheader(
    "🎯 Select Mode"
)

mode = st.radio(
    "Mode:",
    [
        "Single Digit Prediction",
        "Predict All Digits 0–9",
        "Collect Training Data"
    ],
    horizontal=False
)


# ============================================================
# MODE 1
# SINGLE DIGIT PREDICTION
# ============================================================

if mode == "Single Digit Prediction":

    st.header(
        "✍️ Single Digit Prediction"
    )

    st.write(
        "Draw one digit from 0 to 9."
    )

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        canvas_result = st_canvas(

            fill_color="rgba(255,255,255,0)",

            stroke_width=stroke_width,

            stroke_color="#FFFFFF",

            background_color="#000000",

            height=280,

            width=280,

            drawing_mode="freedraw",

            key="single_digit_canvas"
        )

    with col2:

        st.info(
            """
            ### How it works

            1. Draw a digit
            2. Convert to grayscale
            3. Create binary image
            4. Crop the digit
            5. Make it square
            6. Resize to 20 × 20
            7. Add padding
            8. Create 28 × 28 image
            9. Center the digit
            10. Normalize pixels
            11. Send to CNN
            12. Predict digit
            """
        )

    if canvas_result.image_data is not None:

        image_data = canvas_result.image_data

        rgb = image_data[:, :, :3]

        if np.max(rgb) > 10:

            result = preprocess_digit(
                image_data
            )

            if result is not None:

                display_preprocessing(
                    result
                )

                st.divider()

                st.header(
                    "🔍 Final 28 × 28 Pixel Visualization"
                )

                st.write(
                    """
                    Each square represents one pixel.
                    This is the image structure sent to
                    the neural network.
                    """
                )

                fig = show_pixel_grid(
                    result["mnist"]
                )

                st.pyplot(
                    fig
                )

                if model is not None:

                    predicted_digit, confidence, probabilities = \
                        predict_digit(
                            model,
                            result["model_input"]
                        )

                    display_prediction(
                        predicted_digit,
                        confidence,
                        probabilities
                    )


# ============================================================
# MODE 2
# PREDICT ALL DIGITS 0–9
# ============================================================

elif mode == "Predict All Digits 0–9":

    st.header(
        "🔢 Predict All Digits 0–9"
    )

    st.write(
        """
        Draw each digit according to the label shown.
        The application will compare your expected digit
        with the CNN prediction.
        """
    )

    if model is None:

        st.error(
            "CNN model is not available."
        )

    else:

        st.info(
            """
            💡 Draw the requested digit in each box.
            For example, when the box says **Draw 0**,
            draw the digit 0.
            """
        )

        results = {}

        for digit in range(10):

            st.divider()

            st.subheader(
                f"✍️ Draw {digit}"
            )

            col1, col2 = st.columns(
                [1, 2]
            )

            with col1:

                canvas = st_canvas(

                    fill_color="rgba(255,255,255,0)",

                    stroke_width=stroke_width,

                    stroke_color="#FFFFFF",

                    background_color="#000000",

                    height=220,

                    width=220,

                    drawing_mode="freedraw",

                    key=f"all_digit_canvas_{digit}"
                )

            with col2:

                if canvas.image_data is not None:

                    rgb = canvas.image_data[:, :, :3]

                    if np.max(rgb) > 10:

                        result = preprocess_digit(
                            canvas.image_data
                        )

                        if result is not None:

                            predicted, confidence, probabilities = \
                                predict_digit(
                                    model,
                                    result["model_input"]
                                )

                            correct = (
                                predicted == digit
                            )

                            results[digit] = {

                                "expected": digit,

                                "predicted": predicted,

                                "confidence": confidence,

                                "correct": correct
                            }

                            if correct:

                                st.success(
                                    f"Expected: {digit} | "
                                    f"Predicted: {predicted}"
                                )

                            else:

                                st.error(
                                    f"Expected: {digit} | "
                                    f"Predicted: {predicted}"
                                )

                            st.metric(
                                "Confidence",
                                f"{confidence:.2f}%"
                            )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        if len(results) > 0:

            st.divider()

            st.header(
                "📊 0–9 Prediction Summary"
            )

            total = len(results)

            correct_count = sum(
                r["correct"]
                for r in results.values()
            )

            accuracy = (
                correct_count / total * 100
            )

            m1, m2, m3 = st.columns(3)

            with m1:

                st.metric(
                    "Digits Tested",
                    total
                )

            with m2:

                st.metric(
                    "Correct",
                    correct_count
                )

            with m3:

                st.metric(
                    "Accuracy",
                    f"{accuracy:.2f}%"
                )

            summary_data = []

            for digit in range(10):

                if digit in results:

                    r = results[digit]

                    summary_data.append({

                        "Expected":
                            r["expected"],

                        "Predicted":
                            r["predicted"],

                        "Confidence":
                            f"{r['confidence']:.2f}%",

                        "Result":
                            "✅ Correct"
                            if r["correct"]
                            else "❌ Incorrect"
                    })

            if summary_data:

                st.dataframe(
                    summary_data,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# MODE 3
# COLLECT TRAINING DATA
# ============================================================

elif mode == "Collect Training Data":

    st.header(
        "📚 Collect Training Data"
    )

    st.write(
        """
        Use this mode to create your own handwritten
        digit dataset.

        Select the correct label, draw the digit,
        and save the processed 28 × 28 image.
        """
    )

    st.warning(
        """
        ⚠️ The label must match the digit you actually draw.
        For example, if Label = 7, draw the digit 7.
        """
    )

    # --------------------------------------------------------
    # SELECT LABEL
    # --------------------------------------------------------

    selected_label = st.selectbox(
        "Select the correct digit label:",
        list(range(10)),
        key="training_label"
    )

    # ========================================================
    # THREE COLUMN LAYOUT
    # ========================================================

    col1, col2, col3 = st.columns(
        [1, 1, 1],
        gap="small"
    )

    # ========================================================
    # COLUMN 1 — DRAW + CLEAR + SAVE
    # ========================================================

    with col1:

        st.write(
            f"##### Selected Label: **{selected_label}**"
        )

        training_canvas = st_canvas(

            fill_color="rgba(255,255,255,0)",

            stroke_width=stroke_width,

            stroke_color="#FFFFFF",

            background_color="#000000",

            height=280,

            width=280,

            drawing_mode="freedraw",

            display_toolbar=False,

            key=f"training_canvas_{selected_label}_{st.session_state.training_canvas_reset}"
        )

        # --------------------------------------------------------
        # CLEAR CANVAS
        # --------------------------------------------------------

        if st.button("🧹 Clear Canvas"):

            st.session_state.training_canvas_reset += 1

            st.rerun()

        # ----------------------------------------------------
        # PROCESS DRAWING
        # ----------------------------------------------------

        result = None

        if training_canvas.image_data is not None:

            rgb = training_canvas.image_data[:, :, :3]

            if np.max(rgb) > 10:

                result = preprocess_digit(
                    training_canvas.image_data
                )

                if result is not None:

                    # ------------------------------------------------
                    # SAVE BUTTON
                    # ------------------------------------------------

                    if st.button(
                        "💾 Save Training Example",
                        type="primary",
                        #use_container_width=True
                    ):

                        timestamp = datetime.now().strftime(
                            "%Y%m%d_%H%M%S_%f"
                        )

                        label_dir = (
                            TRAINING_DATA_DIR
                            / str(selected_label)
                        )

                        label_dir.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        # --------------------------------------------
                        # SAVE PROCESSED 28 × 28 IMAGE
                        # --------------------------------------------

                        image_path = (
                            label_dir
                            / f"{timestamp}.npy"
                        )

                        np.save(
                            image_path,
                            result["mnist"]
                        )

                        # --------------------------------------------
                        # SAVE METADATA
                        # --------------------------------------------

                        metadata_path = (
                            TRAINING_DATA_DIR
                            / "metadata.csv"
                        )

                        file_exists = (
                            metadata_path.exists()
                        )

                        with open(
                            metadata_path,
                            "a",
                            newline=""
                        ) as f:

                            writer = csv.writer(f)

                            if not file_exists:

                                writer.writerow([
                                    "filename",
                                    "label"
                                ])

                            writer.writerow([
                                str(
                                    image_path.relative_to(
                                        BASE_DIR
                                    )
                                ),
                                selected_label
                            ])

                        st.session_state.collection_count += 1

                        # --------------------------------------------
                        # SUCCESS MESSAGE
                        # --------------------------------------------

                        st.success(
                            f"""
                            ✅ Training example saved!

                            **Label:** {selected_label}

                            **File:** {image_path.name}
                            """
                        )

                        st.info(
                            "Select another digit above "
                            "to start with a clean canvas."
                        )
                else:
                    st.info(
                        "Draw a digit first, then save the "
                        "training example."
                    )

    with col2:
        st.write(
                f"##### ✍️ Draw Digit **{selected_label}**"
            )    
        st.info(
            f"""
                Please draw the digit **{selected_label}** in the canvas.
    
                When you are satisfied with your drawing,
                click **Save Training Example**.
                """
            )
            
        st.info(
                """
                ##### Dataset Collection Process
    
                **Original Drawing**
                ↓    
                Grayscale
                ↓    
                Binary
                ↓    
                Crop
                ↓    
                Square
                ↓    
                20 × 20
                ↓    
                28 × 28
                ↓    
                Save Image + Label
                """
        )


    # --------------------------------------------------------
    # COLUMN 3 — PROCESSED IMAGE PREVIEW ONLY
    # --------------------------------------------------------

    with col3:
                st.write("##### 🔬 Processed Image Preview") 
#        if training_canvas.image_data is not None:
 #           rgb = training_canvas.image_data[:, :, :3]
  #          if np.max(rgb) > 10:
   #             result = preprocess_digit(
   #                 training_canvas.image_data
   #             )
                if result is not None:
                    _, center_col, _ = st.columns([1, 2, 1])
                    with center_col:
                        st.image(
                            result["cropped"],
                            width=130
                        )
                        st.caption("🔹 Cropped")

                    #st.image(
                    #    result["cropped"],
                    #    caption="1️⃣ Cropped",
                    #    clamp=True
                    #)

                        #with c2:
                        st.image(
                            result["resized"],
                            caption="2️⃣ Resized to 20 × 20",
                            clamp=True
                        )

                        #with c3:
                        st.image(
                            result["mnist"],
                            caption="3️⃣ Final MNIST 28 × 28",
                            clamp=True
                        )
                #else:

                #    st.warning(
                #        "Unable to process the drawing."
                #    )
        #     else:
        #         st.info(
        #             "Draw a digit to see the "
        #             "processed images here."
        #         )

        # else:
        #     st.info(
        #         "Draw a digit to see the "
        #         "processed images here."
        #     )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📊 Collected Dataset"
    )

    total_samples = 0

    label_counts = {}

    for digit in range(10):

        label_dir = (
            TRAINING_DATA_DIR
            / str(digit)
        )

        if label_dir.exists():

            count = len(
                list(
                    label_dir.glob("*.npy")
                )
            )

        else:

            count = 0

        label_counts[digit] = count

        total_samples += count

    m1, m2 = st.columns(2)

    with m1:

        st.metric(
            "Total Samples",
            total_samples
        )

    with m2:

        st.metric(
            "Current Session",
            st.session_state.collection_count
        )

    st.dataframe(
        [
            {
                "Digit": digit,
                "Samples": label_counts[digit]
            }

            for digit in range(10)
        ],
        use_container_width=True,
        hide_index=True
    )


    # ============================================================
    # DELETE ALL LEARNING DATA
    # ============================================================

    st.markdown("---")

    st.markdown("##### 🗑️ Manage Learning Data")

    if st.button(
        "🗑️ Delete All Learning Data",
        type="secondary"
    ):
        st.session_state["confirm_delete_training"] = True


    # Warning and confirmation
    if st.session_state.get("confirm_delete_training", False):

        st.warning(
            "⚠️ This will permanently delete ALL collected "
            "training examples. This action cannot be undone."
        )

        confirm_col1, confirm_col2 = st.columns(2)

        with confirm_col1:

            if st.button(
                "❌ Yes, Delete Everything",
                type="primary"
            ):

                training_dir = BASE_DIR / "training_data"

                if training_dir.exists():

                    import shutil

                    # Delete the complete training_data folder
                    shutil.rmtree(training_dir)

                    st.session_state["confirm_delete_training"] = False

                    st.success(
                        "✅ All learning data has been deleted."
                    )

                    # Refresh the application so the
                    # Collected Dataset table is also updated
                    st.rerun()

                else:

                    st.session_state["confirm_delete_training"] = False

                    st.info(
                        "ℹ️ No learning data found."
                    )

        with confirm_col2:

            if st.button("Cancel"):

                st.session_state["confirm_delete_training"] = False

                st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MNIST Digit Recognition | "
    "Streamlit + OpenCV + TensorFlow CNN"
)