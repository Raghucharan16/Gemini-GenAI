import os
import torch
from diffusers import FluxPipeline
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
# Authenticate
token = os.getenv("Hugging_face_api_token")
login(token=token)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    device="cuda:1",
    use_auth_token=True,  # Add this to use your token
)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "bahubali in forest"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")
