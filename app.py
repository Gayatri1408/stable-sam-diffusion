import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import SamPredictor, sam_model_registry

# -------------------------------
# Configuration
# -------------------------------
MODEL_CHECKPOINT = "./checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load SAM
# -------------------------------
print("🔍 Loading SAM...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT)
predictor = SamPredictor(sam)
predictor.model.to(DEVICE)

# -------------------------------
# Load Stable Diffusion Inpainting Pipeline
# -------------------------------
print("🎨 Loading Stable Diffusion Inpainting Pipeline...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float32,
).to(DEVICE)

# -------------------------------
# Global Storage
# -------------------------------
original_image_np = None  # Original image as NumPy array

# -------------------------------
# Image Setup
# -------------------------------
def set_image(image):
    global original_image_np
    image = np.array(image)
    original_image_np = image
    predictor.set_image(image)
    return image

# -------------------------------
# Click-based Mask Generation
# -------------------------------
def segment_from_click(evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
    mask = masks[0]
    mask_image = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask_image)

# -------------------------------
# Inpaint with Prompt
# -------------------------------
def apply_prompt(prompt, mask_image):
    if original_image_np is None or mask_image is None:
        return None

    image = Image.fromarray(original_image_np).convert("RGB")
    mask = mask_image.convert("L").resize(image.size)

    # Run inpainting
    result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    return result

# -------------------------------
# Gradio UI
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Stable SAM Diffusion Editor")
    gr.Markdown("Upload an image → Click to segment → Enter a prompt → Get AI-edited output!")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            prompt_input = gr.Textbox(label="Text Prompt (e.g., 'change shirt to red')")
            mask_output = gr.Image(label="Segmented Mask")
            edited_output = gr.Image(label="Edited Image")

        with gr.Column():
            gr.Markdown("### 🖱️ Click inside image to segment an object")
            interactive = gr.Image(type="numpy", label="Click to Segment").select(
                fn=segment_from_click, inputs=None, outputs=mask_output
            )

    image_input.change(fn=set_image, inputs=image_input, outputs=interactive)
    prompt_input.submit(fn=apply_prompt, inputs=[prompt_input, mask_output], outputs=edited_output)

demo.launch()
