import torch
from diffusers import (
    DiffusionPipeline,
    StableVideoDiffusionPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionControlNetPipeline,
)
from PIL import Image
import numpy as np
import cv2
import gc
import gradio as gr
import time
import os
import imageio

# ====================== SETUP ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"🔥 Using device: {device}")

current_model = {"pipe": None, "name": None, "extra_objs": []}

available_models = {
    "Instruct-Pix2Pix": "timbrooks/instruct-pix2pix",
    "ControlNet": {
        "base": "runwayml/stable-diffusion-v1-5",
        "Canny": "lllyasviel/sd-controlnet-canny",
        "Depth": "lllyasviel/sd-controlnet-depth",
        "HED": "lllyasviel/sd-controlnet-hed",
        "MLSD": "lllyasviel/sd-controlnet-mlsd",
        "Scribble": "lllyasviel/sd-controlnet-scribble"
    },
    "Image-to-Video": "stabilityai/stable-video-diffusion-img2vid-xt",
}

# ====================== HELPER: AUTO-DETECTION ======================
def detect_model_type(img: Image.Image, prompt: str):
    if not prompt or prompt.strip() == "":
        return "Image-to-Video"

    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.sum(edges > 0) / edges.size
    color_var = np.var(arr.reshape(-1, 3))
    contrast = np.std(gray)

    if (edge_ratio > 0.25 and color_var < 2000) or np.mean(gray) < 60 or contrast < 30:
        return "ControlNet"
    else:
        return "Instruct-Pix2Pix"

def classify_controlnet_model(image):
    """
    Classify ControlNet conditioning image realistically into:
    Depth / HED / Scribble / Canny / MLSD
    """

    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    total_pixels = h * w

    # --- Calculate key descriptors ---
    contrast = gray.std()
    mean_val = np.mean(gray)
    unique_colors = len(np.unique(gray))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / total_pixels

    # Binary threshold to detect black–white separation (Canny/Scribble)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    white_ratio = np.sum(binary == 255) / total_pixels

    # Hough transform for line detection (MLSD detection)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=60, maxLineGap=5)
    line_count = 0 if lines is None else len(lines)

    # --- Heuristic decision tree (based on real ControlNet conditions) ---

    # 1️⃣ DEPTH — smooth, grayscale, low edge density
    if contrast < 35 and edge_density < 0.01 and unique_colors > 100:
        return "Depth"

    # 2️⃣ CANNY — very high edge density, binary pattern (strong black-white)
    if edge_density > 0.05 and unique_colors < 100 and (white_ratio < 0.6 or white_ratio > 0.9):
        return "Canny"

    # 3️⃣ MLSD — many straight lines, high line count
    if line_count > 60 and edge_density > 0.02 and contrast > 40:
        return "MLSD"

    # 4️⃣ HED — soft edges, moderate contrast, not binary
    if 0.01 <= edge_density <= 0.05 and 100 <= unique_colors <= 200 and contrast > 35:
        return "HED"

    # 5️⃣ SCRIBBLE — high unique colors (drawn strokes), often inverted edges
    if unique_colors > 180 and contrast > 25 and edge_density > 0.03:
        return "Scribble"

    # Fallback
    return "Depth"

# ====================== PREPROCESS ======================
def preprocess_controlnet_input(input_image: Image.Image, variant: str):
    input_image = input_image.resize((512, 512))
    img = np.array(input_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if variant == "Canny":
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
    elif variant == "Scribble":
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        return Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
    else:
        return input_image

# ====================== ENHANCER (Post-Processing) ======================
def enhance_image(img_pil: Image.Image):
    img = np.array(img_pil)
    img = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=5)
    return Image.fromarray(img)

# ====================== MODEL LOADING ======================
def load_model(model_type, input_image=None):
    global current_model

    # Cached reuse
    if current_model["pipe"] is not None and model_type in current_model["name"]:
        return f"{current_model['name']} (cached)"

    # Cleanup previous
    if current_model["pipe"]:
        try:
            current_model["pipe"].to("cpu")
        except:
            pass
        del current_model["pipe"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        current_model = {"pipe": None, "name": None, "extra_objs": []}

    # Load appropriate model
    if model_type == "ControlNet":
        variant = detect_controlnet_variant(input_image)
        controlnet = ControlNetModel.from_pretrained(
            available_models["ControlNet"][variant],
            torch_dtype=dtype
        ).to(device)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            available_models["ControlNet"]["base"],
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=dtype
        ).to(device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        current_model = {"pipe": pipe, "name": f"ControlNet-{variant}", "extra_objs": [controlnet]}

    elif model_type == "Instruct-Pix2Pix":
        pipe = DiffusionPipeline.from_pretrained(
            available_models["Instruct-Pix2Pix"],
            torch_dtype=dtype,
            safety_checker=None
        ).to(device)
        pipe.enable_model_cpu_offload()
        current_model = {"pipe": pipe, "name": "Instruct-Pix2Pix", "extra_objs": []}

    elif model_type == "Image-to-Video":
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            available_models["Image-to-Video"],
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        pipe.enable_model_cpu_offload()
        current_model = {"pipe": pipe, "name": "Image-to-Video", "extra_objs": []}

    return f"{current_model['name']} loaded ✅"

# ====================== GENERATION ======================
def generate(prompt, input_image):
    if input_image is None:
        return None, None, "⚠ Please upload an image."
        input_image = input_image.resize((512, 512))

    model_type = detect_model_type(input_image, prompt)
    status = load_model(model_type, input_image)
    pipe = current_model["pipe"]
    start = time.time()

    if model_type == "ControlNet":
        variant = current_model["name"].split("-")[1]
        processed = preprocess_controlnet_input(input_image, variant)
        with torch.inference_mode(), torch.autocast(device):
            out = pipe(prompt=prompt, image=processed, num_inference_steps=20, guidance_scale=7.5).images[0]
        out = enhance_image(out)
        return out, None, f"✅ {model_type}-{variant} | {status} | ⏱ {time.time() - start:.2f}s"

    elif model_type == "Instruct-Pix2Pix":
        with torch.inference_mode(), torch.autocast(device):
            out = pipe(prompt=prompt, image=input_image, strength=0.9, num_inference_steps=20, guidance_scale=7.5).images[0]
        out = enhance_image(out)
        return out, None, f"✅ {model_type} | {status} | ⏱ {time.time() - start:.2f}s"

    else:  # Image-to-Video
        input_image = input_image.resize((576, 576))
        with torch.inference_mode(), torch.autocast(device):
            result = pipe(input_image, decode_chunk_size=4, motion_bucket_id=127, noise_aug_strength=0.02)
        frames = result.frames[0]
        os.makedirs("outputs", exist_ok=True)
        video_path = "outputs/generated_video.mp4"
        imageio.mimsave(video_path, [np.array(f) for f in frames], fps=6)
        return None, video_path, f"🎥 Generated Video | {status} | ⏱ {time.time() - start:.2f}s"



def create_vibrant_interface():
    """Defines and launches the Gradio web interface with a vibrant theme."""

    # Define a custom, vibrant theme (Monochrome base with a bright primary color)
    VIBRANT_TEAL = "#00bcd4" # A bright, energetic teal/cyan

    custom_theme = gr.themes.Monochrome(
        primary_hue=gr.themes.Color(
            name="teal",
            c50="#e0f7fa",
            c100="#b2ebf2",
            c200="#80deea",
            c300="#4dd0e1",
            c400="#26c6da",
            c500=VIBRANT_TEAL,
            c600="#00acc1",
            c700="#0097a7",
            c800="#00838f",
            c900="#006064",
            c950="#004d40"
        ),
        secondary_hue="gray",
    ).set(
        # General background and foreground colors
        body_background_fill='black',
        body_background_fill_dark='black',
        block_background_fill='#1a1a1a', # Dark grey for contrast blocks
        color_accent_soft='rgba(0, 188, 212, 0.2)',

        # Button styling (Primary is bright teal)
        button_primary_background_fill=VIBRANT_TEAL,
        button_primary_background_fill_hover='#00e5ff',
        button_primary_text_color='black', # Black text on bright button

        # Text/Header colors
        # text_color='white', # Removed this argument
        block_title_text_color=VIBRANT_TEAL,

        # Shadow/Border for modern look
        shadow_drop_lg='0 10px 15px rgba(0, 0, 0, 0.5)',
    )

    with gr.Blocks(title="Vibrant Diffusion Creator", theme=custom_theme) as demo:
        # --- Title Block ---
        with gr.Row(variant="panel"):
            gr.Markdown(
                f"""
                # NovaGAN
                """,
                # Custom CSS for the title area if needed, but the theme handles most of it
            )

        # --- Main Layout ---
        with gr.Row(equal_height=True):
            # --- Input Column (Left) ---
            with gr.Column(scale=1, variant="box"):
                gr.Markdown("## 💡 Input & Instruction")

                image_input = gr.Image(
                    type="pil",
                    label="1. Upload Image (Required)",
                    height=300,
                    interactive=True
                )

                prompt_input = gr.Textbox(
                    label="2. Prompt / Text Instruction",
                    placeholder="E.g., 'A vibrant neon city' (for ControlNet) or 'Change the car color to yellow' (for Pix2Pix)",
                    lines=3
                )

                generate_btn = gr.Button(
                    "🚀 GENERATE",
                    variant="primary",
                    size="lg"
                )

            # --- Output Column (Right) ---
            with gr.Column(scale=1, variant="box"):
                gr.Markdown("## ✨ Result & Status")

                # Use a Tabbed Interface for cleaner output switching
                with gr.Tabs():
                    with gr.TabItem("🖼 Image Output"):
                        image_output = gr.Image(
                            type="pil",
                            label="Generated Image",
                            height=300,
                            show_download_button=True
                        )

                    with gr.TabItem("🎥 Video Output"):
                        video_output = gr.Video(
                            label="Generated Video (Image-to-Video)",
                            height=300,
                        )

                status_output = gr.Textbox(
                    label="🔥 Status / Model Used",
                    interactive=False,
                    # color=VIBRANT_TEAL, # Removed this argument
                    autofocus=False
                )


        # --- Event Handlers ---
        # The generate function returns: (Image, VideoPath, StatusText)
        generate_btn.click(
            fn=generate,
            inputs=[prompt_input, image_input],
            outputs=[image_output, video_output, status_output]
        )

        # --- Footer ---
        gr.Markdown(
            """
            ---
            **Auto-Detection Logic:** No Prompt = Image-to-Video.
            """
        )

    # Launch the app
    print("\n✅ Gradio app launching with a vibrant, website-like theme.")
    demo.launch(share=True)

# Run the interface creation function
if _name_ == "_main_":
    # You must call the new function name here
    create_vibrant_interface()
