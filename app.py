from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from inference import ChineseCharacterRecognizer


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_OPTIONS = {
    "Improved CNN": MODELS_DIR / "improved_cnn.pth",
    "Simple CNN": MODELS_DIR / "simple_cnn_5fold.pth",
    "ResNet18": MODELS_DIR / "resnet18.pth",
    "LW-ViT": MODELS_DIR / "lwvit.pth",
    "PF-ViT": MODELS_DIR / "pfvit.pth"
}


@st.cache_resource
def load_recognizer(model_name, model_path):
    return ChineseCharacterRecognizer(
        model_name=model_name,
        model_path=model_path
    )


@st.cache_data
def load_comparison_table():
    comparison_path = RESULTS_DIR / "model_comparison_summary.csv"

    if not comparison_path.exists():
        return None

    table = pd.read_csv(comparison_path)

    required_columns = [
        "model_name",
        "test_accuracy",
        "test_f1",
        "number_of_parameters",
        "model_size_mb"
    ]

    available_columns = [
        column for column in required_columns
        if column in table.columns
    ]

    return table[available_columns]


def get_selected_model_metrics(model_name):
    table = load_comparison_table()

    if table is None or "model_name" not in table.columns:
        return None

    matched_rows = table[table["model_name"] == model_name]

    if matched_rows.empty:
        return None

    return matched_rows.iloc[0]


def format_metric(value):
    if pd.isna(value):
        return "-"

    return f"{value:.4f}"


st.set_page_config(
    page_title="Handwritten Chinese Character Recognition",
    page_icon="字",
    layout="wide"
)

st.title("Handwritten Chinese Character Recognition")

st.write(
    "This software module recognizes handwritten Chinese characters using trained deep learning models. "
    "The models were trained on the CASIA_246 dataset with 246 handwritten Chinese character classes."
)

with st.sidebar:
    st.header("Model Selection")

    model_name = st.selectbox(
        "Choose a trained model",
        options=list(MODEL_OPTIONS.keys()),
        index=0
    )

    model_path = MODEL_OPTIONS[model_name]

    st.subheader("Selected Model")

    st.write("Model:", model_name)
    st.write("Checkpoint:", str(model_path.relative_to(BASE_DIR)))

    if not model_path.exists():
        st.error("The selected model checkpoint was not found.")

    model_metrics = get_selected_model_metrics(model_name)

    if model_metrics is not None:
        if "test_accuracy" in model_metrics:
            st.write("Test accuracy:", format_metric(model_metrics["test_accuracy"]))

        if "test_f1" in model_metrics:
            st.write("Test macro F1-score:", format_metric(model_metrics["test_f1"]))

        if "number_of_parameters" in model_metrics:
            st.write("Parameters:", f"{int(model_metrics['number_of_parameters']):,}")

        if "model_size_mb" in model_metrics:
            st.write("Model size:", f"{model_metrics['model_size_mb']:.2f} MB")

st.header("Upload Character Image")

uploaded_file = st.file_uploader(
    "Upload an image of a handwritten Chinese character",
    type=["png", "jpg", "jpeg"]
)

left_column, right_column = st.columns([1, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with left_column:
        st.subheader("Uploaded Image")
        st.image(image, caption="Input image", width="stretch")

    recognize_button = st.button("Recognize Character", type="primary")

    if recognize_button:
        try:
            recognizer = load_recognizer(model_name, str(model_path))
            result = recognizer.predict(image, top_k=5)
            checkpoint_metrics = recognizer.get_checkpoint_metrics()

            with right_column:
                st.subheader("Prediction Result")

                st.markdown(
                    f"""
                    <div style="font-size: 96px; text-align: center; font-weight: 600;">
                        {result["predicted_character"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.metric(
                    label="Confidence",
                    value=f"{result['confidence'] * 100:.2f}%"
                )

                st.write("Top-5 predictions")

                top_predictions_df = pd.DataFrame([
                    {
                        "Rank": index + 1,
                        "Character": item["character"],
                        "Class Index": item["class_index"],
                        "Confidence": f"{item['confidence'] * 100:.2f}%"
                    }
                    for index, item in enumerate(result["top_predictions"])
                ])

                st.dataframe(top_predictions_df, width="stretch", hide_index=True)

                st.write("Model information")

                info_rows = {
                    "Model": model_name,
                    "Image size": checkpoint_metrics["image_size"],
                    "Number of classes": checkpoint_metrics["num_classes"],
                    "Best validation accuracy": checkpoint_metrics["best_validation_accuracy"],
                    "Best validation F1-score": checkpoint_metrics["best_validation_f1"]
                }

                info_df = pd.DataFrame(
                    [{"Metric": key, "Value": str(value)} for key, value in info_rows.items()]
                )

                st.dataframe(info_df, width="stretch", hide_index=True)

        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)

else:
    st.info("Upload a handwritten Chinese character image to start recognition.")

st.header("Model Comparison")

comparison_table = load_comparison_table()

if comparison_table is not None:
    display_table = comparison_table.copy()

    if "test_accuracy" in display_table.columns:
        display_table["test_accuracy"] = display_table["test_accuracy"].round(4)

    if "test_f1" in display_table.columns:
        display_table["test_f1"] = display_table["test_f1"].round(4)

    if "model_size_mb" in display_table.columns:
        display_table["model_size_mb"] = display_table["model_size_mb"].round(2)

    st.dataframe(display_table, width="stretch", hide_index=True)
else:
    st.warning("Model comparison file was not found.")
