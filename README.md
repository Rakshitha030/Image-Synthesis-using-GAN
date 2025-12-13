# Image-Synthesis-using-GAN
This project (NovaGAN) is a smart, all-in-one diffusion-based image & video generation app built using PyTorch + Hugging Face Diffusers + Gradio.

What it does:

📸 Takes an input image

✍️ Reads a prompt (optional)

🧠 Automatically decides which model to use:

Instruct-Pix2Pix → for editing images using text instructions

ControlNet → for structure-aware generation (Canny, Depth, HED, Scribble, MLSD)

Image-to-Video → if no prompt is given, converts an image into a short video

⚙️ Applies image preprocessing + enhancement

🌐 Launches a modern Gradio web UI with image & video tabs

Key highlights:

Auto model detection (no manual switching)

Automatic ControlNet variant classification

GPU-optimized (FP16, autocast, CPU offload)

Clean, dark, vibrant UI

Outputs both images and videos

```mermaid
flowchart TD
  A --> B

