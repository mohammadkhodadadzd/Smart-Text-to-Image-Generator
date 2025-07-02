import gradio as gr
import torch
from diffusers import AutoPipelineForText2Image
import gc

# --- 1. Initial Setup ---

# Detect device (GPU if available, otherwise CPU)
# Warning: This will be very slow on CPU!
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {torch_dtype}")

# Dictionary to hold loaded models (model caching)
pipelines = {}

# Model IDs on Hugging Face
MODEL_IDS = {
    "Stable Diffusion 1.5": "runwayml/stable-diffusion-v1-5",
    "SDXL 1.0": "stabilityai/stable-diffusion-xl-base-1.0"
}

# --- 2. Model loading function (with smart caching) ---

def load_pipeline(model_name):
    """
    Loads the selected model. If the model is already loaded, uses the cache.
    """
    global pipelines
    if model_name in pipelines:
        print(f"Model '{model_name}' already in cache. Using cached pipeline.")
        return pipelines[model_name]

    # If a new model is selected, clear previous models from memory
    if pipelines:
        print("Clearing old models from memory...")
        pipelines.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_id = MODEL_IDS[model_name]
    print(f"Loading model: {model_name} ({model_id})... This may take a few minutes.")

    # Load pipeline with optimized settings for GPU
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True
    ).to(device)

    # Store in cache
    pipelines[model_name] = pipe
    print(f"Model '{model_name}' loaded successfully.")
    return pipe

# --- 3. Main function for image generation ---

def generate_image(prompt, model_choice, cfg_scale, width, height):
    """
    Generates an image based on user inputs.
    """
    if not prompt:
        raise gr.Error("Please enter a prompt!")

    try:
        # Get the pipeline (from cache or load a new one)
        pipe = load_pipeline(model_choice)

        print("Generating image...")
        # Generate the image
        image = pipe(
            prompt=prompt,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            num_inference_steps=30 # Number of steps for better quality
        ).images[0]

        print("Image generated successfully.")
        return image

    except Exception as e:
        print(f"An error occurred: {e}")
        # Clear cache in case of error (e.g., out of memory)
        if pipelines:
            pipelines.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        raise gr.Error(f"An error occurred: {e}")

# --- 4. Build Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as demo:
    gr.Markdown("# 🖼️ Smart application for generating images from text")
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            prompt = gr.Textbox(
                label="Prompt",
                info="Enter a text description of the image you want to create:",
                placeholder="For example: A majestic lion jumping from a big stone at sunset"
            )

            model_choice = gr.Radio(
                choices=list(MODEL_IDS.keys()),
                value="SDXL 1.0",
                label="Model",
                info="SDXL has higher quality but requires more memory."
            )

            cfg_scale = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                step=0.5,
                value=7.5,
                label="CFG",
                info="Higher values result in more prompt compliance."
            )

            with gr.Row():
                width = gr.Slider(
                    minimum=512,
                    maximum=1024,
                    step=64,
                    value=1024,
                    label="Width"
                )
                height = gr.Slider(
                    minimum=512,
                    maximum=1024,
                    step=64,
                    value=1024,
                    label="Height"
                )

            submit_btn = gr.Button("Generate Image", variant="primary")

        with gr.Column(scale=1):

            output_image = gr.Image(label="Output Image")
            gr.Markdown(
                """
                **Guide:**
                - **SD 1.5** performs best at `512x512` dimensions.
                - **SDXL 1.0** performs best at `1024x1024` dimensions.
                - Loading the model for the first time may take a few minutes.
                """
            )

    # Connect the button to the main function
    submit_btn.click(
        fn=generate_image,
        inputs=[prompt, model_choice, cfg_scale, width, height],
        outputs=output_image
    )

if __name__ == "__main__":
    load_pipeline("SDXL 1.0")
    demo.launch(share=True)