"""
Gradio UI for the Nature Scene Classifier.

This file only handles the user interface and delegates prediction logic
to core.predict.predict_image_pil(), following clean architecture principles.
"""

from fastai.vision.all import PILImage
import gradio as gr
from core.predict import predict_image_pil
from typing import Dict, Any


# --------------------------------------------------------------------------------------
# Wrapper for Gradio prediction
# --------------------------------------------------------------------------------------
def gradio_predict(img) -> Dict[str, float]:
    """
    Wrapper used by Gradio. Ensures the UI never crashes due to invalid input.

    Parameters
    ----------
    img : PIL.Image or ndarray
        Image uploaded by the user.

    Returns
    -------
    Dict[str, float]
        Dictionary of class probabilities. If input is invalid, returns {"error": 1.0}.
    """
    if img is None:
        return {"error": 1.0}

    try:
        return predict_image_pil(img)
    except Exception as e:
        # You can print(e) or log it if desired.
        return {"error": 1.0}


# --------------------------------------------------------------------------------------
# BUILD UI WITH GRADIO BLOCKS (compatible with older Gradio versions)
# --------------------------------------------------------------------------------------
with gr.Blocks(title="Nature Scene Classifier") as demo:

    # -----------------------------------------------------
    # GLOBAL CSS (legacy-compatible)
    # -----------------------------------------------------
    gr.HTML("""
    <style>
        body {
            background-color: #eef5f0 !important;
            font-family: 'Segoe UI', sans-serif;
        }
        .custom-box {
            background: #ffffffdd;
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 14px;
            box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
        }
        .custom-title {
            color: #2e4f3e;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 6px;
        }
        .custom-sub {
            color: #3b5f4a;
            font-size: 18px;
            margin-bottom: 14px;
        }
        .gr-button {
            background-color: #4a7c59 !important;
            color: white !important;
            border-radius: 10px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            padding: 10px 20px !important;
            border: none !important;
        }
    </style>
    """)

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------
    gr.HTML("""
    <div class="custom-box">
        <div class="custom-title">🌿 Nature Scene Classifier</div>
        <div class="custom-sub">
            Upload an image and the model will classify it into:
            <b>forest · beach · bird · fish</b>
        </div>
    </div>
    """)

    # -----------------------------------------------------
    # MAIN INTERFACE
    # -----------------------------------------------------
    with gr.Row():

        # LEFT COLUMN — IMAGE UPLOAD
        with gr.Column(scale=1):
            gr.HTML("<div class='custom-box'>")
            image_input = gr.Image(
                type="pil",
                label="Upload an image",
                height=300
            )
            submit_btn = gr.Button("🔍 Classify")
            gr.HTML("</div>")

        # RIGHT COLUMN — RESULTS
        with gr.Column(scale=1):
            gr.HTML("<div class='custom-box'>")
            label_output = gr.Label(
                num_top_classes=4,
                label="Prediction"
            )
            gr.HTML("</div>")

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------
    gr.HTML("""
    <div class="custom-box">
        <p><b>🧠 Model details:</b><br>
        Built using <b>FastAI + PyTorch</b>, packaged with clean architecture practices,
        and deployed on <b>HuggingFace Spaces</b>.</p>

        <p>👨‍💻 <b>Created by:</b> Nicolás A<br>
        🔗 GitHub: <a href="https://github.com/nicoacosta04" target="_blank">@nicoacosta04</a></p>
    </div>
    """)

    # -----------------------------------------------------
    # BUTTON CALLBACK
    # -----------------------------------------------------
    submit_btn.click(
        fn=gradio_predict,
        inputs=image_input,
        outputs=label_output
    )


# --------------------------------------------------------------------------------------
# LAUNCH APP
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()