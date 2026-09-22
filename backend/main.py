"""
Image Quality Enhancement - Backend API

Uses:
- Fine-tuned Real-ESRGAN model for smaller images (<= 350 px)
- Pretrained Real-ESRGAN model for larger images (> 350 px)

Both models run locally on CPU.

NOTE: To fit within limited server memory (free hosting tiers usually
give 512MB), only ONE model is kept loaded in memory at a time. It's
loaded the first time it's needed, and swapped out if a request needs
the other one. This trades a little speed (occasionally reloading a
model) for a much smaller memory footprint.
"""

import gc
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


# PATHS


# Project structure:
#
# project/
# ├── backend/
# │   ├── main.py
# │   └── models/
# │       ├── realesrgan_finetuned_2000.pth
# │       └── RealESRGAN_x4plus.pth
# ├── BasicSR/
# └── Real-ESRGAN/

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASICSR_PATH = PROJECT_ROOT / "BasicSR"
MODELS_PATH = Path(__file__).resolve().parent / "models"

# Add BasicSR to Python's import path
sys.path.insert(0, str(BASICSR_PATH))

from basicsr.archs.rrdbnet_arch import RRDBNet



# SETTINGS


DEVICE = torch.device("cpu")

THRESHOLD = 350

FT_MODEL_PATH = MODELS_PATH / "realesrgan_finetuned_2000.pth"
PRETRAINED_MODEL_PATH = MODELS_PATH / "RealESRGAN_x4plus.pth"



# MODEL LOADING


def create_model():
    """Create the RRDBNet architecture used by both checkpoints."""

    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )


def load_model(model_path: Path):
    """Load an RRDBNet model from a checkpoint."""

    print(f"Loading model: {model_path.name}")

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    # Your fine-tuned checkpoint uses "params"
    # Your pretrained checkpoint uses "params_ema"
    if "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif "params" in checkpoint:
        state_dict = checkpoint["params"]
    else:
        state_dict = checkpoint

    model = create_model()

    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    print(f"Loaded successfully: {model_path.name}")

    return model


# --------------------------------------------------------------
# Lazy model cache: only one model lives in memory at a time.
# --------------------------------------------------------------
_current_model = None
_current_model_name = None


def get_model(name: str):
    """
    Return the requested model, loading it if needed.
    If a different model is currently loaded, it is released
    from memory first to keep peak memory usage low.
    """
    global _current_model, _current_model_name

    if _current_model_name == name:
        return _current_model

    # A different model is loaded (or none yet) -- free it first
    if _current_model is not None:
        print(f"Releasing model from memory: {_current_model_name}")
        del _current_model
        gc.collect()

    if name == "finetuned":
        _current_model = load_model(FT_MODEL_PATH)
    else:
        _current_model = load_model(PRETRAINED_MODEL_PATH)

    _current_model_name = name

    return _current_model


print("=" * 60)
print("Backend starting -- models will load on first use")
print("(kept lazy to fit within limited server memory)")
print("=" * 60)



# FASTAPI


app = FastAPI(title="Image Quality Enhancement API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# IMAGE ENHANCEMENT

def enhance_image(image: Image.Image) -> Image.Image:
    """
    Enhance an image using Real-ESRGAN.

    Model selection:
        smaller side <= 350 px  -> fine-tuned model
        smaller side > 350 px   -> pretrained model
    """

    width, height = image.size
    smaller_side = min(width, height)

    # Select model
    if smaller_side <= THRESHOLD:
        model = get_model("finetuned")
        model_name = "Fine-tuned Real-ESRGAN"
    else:
        model = get_model("pretrained")
        model_name = "Pretrained Real-ESRGAN"

    print(
        f"Enhancing {width}x{height} image "
        f"using {model_name}"
    )

    # PIL -> NumPy
    image_np = np.array(image).astype(np.float32) / 255.0

    # HWC -> CHW
    image_tensor = torch.from_numpy(
        image_np.transpose(2, 0, 1)
    ).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    # Run model
    with torch.no_grad():
        output_tensor = model(image_tensor)

    # Tensor -> NumPy
    output_np = (
        output_tensor.squeeze(0)
        .cpu()
        .clamp(0, 1)
        .numpy()
    )

    # CHW -> HWC
    output_np = output_np.transpose(1, 2, 0)

    # Float -> uint8
    output_np = (output_np * 255.0).round().astype(np.uint8)

    # NumPy -> PIL
    output_image = Image.fromarray(output_np)

    print(f"Output size: {output_image.size}")

    return output_image

# API ROUTES

@app.get("/")
def read_root():
    return {
        "status": "Backend is running",
        "message": "Real-ESRGAN enhancement API"
    }


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    """
    Receive an image, enhance it with Real-ESRGAN,
    and return the enhanced PNG.
    """

    contents = await file.read()

    input_image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    output_image = enhance_image(input_image)

    buffer = io.BytesIO()

    output_image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png"
    )
