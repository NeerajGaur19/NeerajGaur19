import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import csv
import plotly.graph_objects as go
import time

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

def preprocess_digit(image, progress_callback=None):

    def report_progress(step):
        if progress_callback is not None:
            progress_callback(step)

    # --------------------------------------------------------
    # 1. RGBA → RGB
    # --------------------------------------------------------

    rgb_image = image[:, :, :3]
    report_progress(1)

    # --------------------------------------------------------
    # 2. RGB → GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY
    )

    report_progress(2)

    # --------------------------------------------------------
    # 3. BINARY IMAGE
    # --------------------------------------------------------

    _, binary = cv2.threshold(
        gray,
        50,
        255,
        cv2.THRESH_BINARY
    )

    report_progress(3)

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

    report_progress(4)

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

    report_progress(5)

    # --------------------------------------------------------
    # 7. RESIZE TO 20 × 20
    # --------------------------------------------------------

    resized_digit = cv2.resize(
        square,
        (20, 20),
        interpolation=cv2.INTER_AREA
    )

    report_progress(6)

    # --------------------------------------------------------
    # 8. CREATE 28 × 28 MNIST IMAGE
    # --------------------------------------------------------

    mnist_before_center = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    report_progress(7)

    mnist_before_center[
        4:24,
        4:24
    ] = resized_digit

    report_progress(8)

    # --------------------------------------------------------
    # 9. CENTER BY MASS
    # --------------------------------------------------------

    mnist_image = center_by_mass(
        mnist_before_center
    )

    report_progress(9)

    # --------------------------------------------------------
    # 10. NORMALIZE
    # --------------------------------------------------------

    normalized = (
        mnist_image.astype("float32") / 255.0
    )

    report_progress(10)

    # --------------------------------------------------------
    # 11. CNN INPUT
    # --------------------------------------------------------

    model_input = normalized.reshape(
        1,
        28,
        28,
        1
    )

    report_progress(11)

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
        figsize=(3, 3)
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


# ============================================================
# FLOWCHART BOX
# ============================================================

def render_flowchart_box(
    placeholder,
    step_number,
    label,
    active_step=0,
    all_completed=False,
    direction="right",
):
    """Render a flowchart stage with direction-specific chevrons."""

    if all_completed or step_number < active_step:
        box_color = "#16A34A"
    elif step_number == active_step:
        box_color = "#2563EB"
    else:
        box_color = "#475569"

    # Point the chevron in the direction of processing.
    polygons = {
        "right": "0% 0%, 86% 0%, 100% 50%, 86% 100%, 0% 100%",
        "left": "14% 0%, 100% 0%, 100% 100%, 14% 100%, 0% 50%",
        "down": "0% 0%, 100% 0%, 100% 86%, 50% 100%, 0% 86%",
        "up": "0% 14%, 50% 0%, 100% 14%, 100% 100%, 0% 100%",
    }
    polygon = polygons.get(direction, polygons["right"])
    is_vertical = direction in ("up", "down")
    min_height = "120px" if is_vertical else "64px"
    padding = "12px 7px" if is_vertical else "8px 18px 8px 8px"
    font_size = "13px" if is_vertical else "14px"

    chevron = f"""
    <div style="
        width: 100%;
        min-height: {min_height};
        box-sizing: border-box;
        background: {box_color};
        color: white;
        border: 1px solid #E2E8F0;
        box-shadow: 3px 3px 0 #172033;
        clip-path: polygon({polygon});
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: {padding};
        font-size: {font_size};
        font-weight: 700;
        overflow-wrap: anywhere;
    ">
        {step_number}. {label}
    </div>
    """
    placeholder.markdown(chevron, unsafe_allow_html=True)


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
            width=120
        )

    with c2:

        st.subheader(
            "2️⃣ Binary Image"
        )

        st.image(
            result["binary"],
            clamp=True,
            width=120
        )

    with c3:
        st.subheader(
            "3️⃣ Cropped Digit"
        )

        st.image(
            result["cropped"],
            clamp=True,
            width=120
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
            width=120
        )

    with c5:

        st.subheader(
            "5️⃣ Reduced to 20 × 20"
        )

        st.image(
            result["resized"],
            clamp=True,
            width=120
        )

    with c6:

        st.subheader(
            "6️⃣ Final MNIST 28 × 28"
        )

        st.image(
            result["mnist"],
            clamp=True,
            width=120
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

    st.header("✍️ Single Digit Prediction")

    # --------------------------------------------------------
    # FLOWCHART STAGE DEFINITIONS
    # --------------------------------------------------------

    flowchart_steps = [
        "RGB conversion", "Grayscale", "Binary image",
        "Crop digit", "Make square", "Resize 20x20",
        "Add padding", "Create 28x28", "Center digit",
        "Normalize", "Send to CNN", "Predict digit",
    ]

    flowchart_boxes = {}

    def add_flowchart_box(step_number, label, direction="right"):
        box_placeholder = st.empty()
        flowchart_boxes[step_number] = {
            "placeholder": box_placeholder,
            "label": label,
            "direction": direction,
        }
        render_flowchart_box(
            box_placeholder, step_number, label,
            active_step=0, all_completed=False,
            direction=direction,
        )

    def update_flowchart(step, completed=False):
        for number, box in flowchart_boxes.items():
            render_flowchart_box(
                box["placeholder"], number, box["label"],
                active_step=step, all_completed=completed,
                direction=box["direction"],
            )
        time.sleep(0.5)

    # TOP: 1 -> 2 -> 3
    top1, top2, top3 = st.columns(3, gap="medium")
    with top1:
        add_flowchart_box(1, flowchart_steps[0], "right")
    with top2:
        add_flowchart_box(2, flowchart_steps[1], "right")
    with top3:
        add_flowchart_box(3, flowchart_steps[2], "right")

    # MIDDLE: symmetric side columns keep the canvas centred.
    # Wider side columns give stages 4-6 and 10-12 more room.
    left_col, centre_col, right_col = st.columns(
        [1.6, 4.8, 1.6], gap="small"
    )

    with left_col:
        # Stages are displayed top-to-bottom as 12, 11, 10,
        # so the processing direction runs upward: 10 -> 11 -> 12.
        add_flowchart_box(12, flowchart_steps[11], "up")
        add_flowchart_box(11, flowchart_steps[10], "up")
        add_flowchart_box(10, flowchart_steps[9], "up")

    with centre_col:
        st.markdown("<div style='text-align:center'>Draw one digit from 0 to 9.</div>", unsafe_allow_html=True)
        _, canvas_center, _ = st.columns([1, 2, 1], gap="small")
        with canvas_center:
            canvas_result = st_canvas(
                fill_color="rgba(255,255,255,0)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="single_digit_canvas",
            )

    with right_col:
        add_flowchart_box(4, flowchart_steps[3], "down")
        add_flowchart_box(5, flowchart_steps[4], "down")
        add_flowchart_box(6, flowchart_steps[5], "down")

    # BOTTOM: stages 7, 8, 9 are positioned right-to-left so that
    # the visual processing direction is 6 -> 7 -> 8 -> 9.
    bottom1, bottom2, bottom3 = st.columns(3, gap="medium")
    with bottom1:
        add_flowchart_box(9, flowchart_steps[8], "left")
    with bottom2:
        add_flowchart_box(8, flowchart_steps[7], "left")
    with bottom3:
        add_flowchart_box(7, flowchart_steps[6], "left")

    # --------------------------------------------------------
    # PREPROCESS DRAWING AND PREDICT
    # --------------------------------------------------------

    if canvas_result.image_data is not None:

        image_data = canvas_result.image_data
        rgb = image_data[:, :, :3]

        if np.max(rgb) > 10:

            result = preprocess_digit(
                image_data,
                progress_callback=update_flowchart,
            )

            if result is not None:
                display_preprocessing(result)
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

                fig = show_pixel_grid(result["mnist"])
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

                if model is not None:
                    update_flowchart(12)
                    predicted_digit, confidence, probabilities = \
                        predict_digit(
                            model,
                            result["model_input"],
                        )

                    # Mark every stage as completed.
                    update_flowchart(12, completed=True)
                    display_prediction(
                        predicted_digit,
                        confidence,
                        probabilities,
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