"""The local Gradio interface uses the same frozen model recipe as the Space."""

from functools import lru_cache
import os


def create_demo():
    import gradio as gr

    @lru_cache(maxsize=1)
    def predictor():
        from .assets import ensure_model_assets
        from .inference import MrootEnsemble
        return MrootEnsemble(ensure_model_assets(), device=os.environ.get("MROOT_DEVICE", "cpu"))

    def classify(image):
        if image is None:
            raise gr.Error("Upload a plant photo first.")
        return predictor().predict(image)

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Plant photo", sources=["upload"]),
        outputs=gr.Label(num_top_classes=5, label="Most likely diseases"),
        title="MROOT Plant Disease Classifier",
        description="Upload a plant photo. MROOT recognizes 39 disease categories and does not include a healthy category.",
        submit_btn="Analyze photo",
        clear_btn="Clear",
        flagging_mode="never",
        analytics_enabled=False,
        concurrency_limit=1,
        api_name="predict",
        delete_cache=(3600, 3600),
    )
