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

flowchart TD
    A[User Uploads Image] --> B[Prompt Provided?]

    B -- No --> C[Image-to-Video Pipeline]
    C --> C1[Stable Video Diffusion]
    C1 --> C2[Generate Video Frames]
    C2 --> C3[Save MP4 Video]

    B -- Yes --> D[Image Analysis Module]
    D --> D1[Edge Density]
    D --> D2[Contrast & Color Variance]
    D --> D3[Auto Model Selection]

    D3 --> E{Selected Model}

    E -->|Instruct-Pix2Pix| F[Text-based Image Editing]
    F --> F1[Pix2Pix Diffusion]
    F1 --> F2[Image Enhancement]

    E -->|ControlNet| G[ControlNet Variant Classifier]
    G --> G1[Canny]
    G --> G2[Depth]
    G --> G3[HED]
    G --> G4[MLSD]
    G --> G5[Scribble]

    G1 --> H[Preprocess Conditioning Image]
    G2 --> H
    G3 --> H
    G4 --> H
    G5 --> H

    H --> I[Stable Diffusion + ControlNet]
    I --> I1[Guided Image Generation]
    I1 --> I2[Post-processing Enhancement]

    F2 --> J[Final Image Output]
    I2 --> J
    C3 --> K[Final Video Output]

    J --> L[Gradio UI Display]
    K --> L
