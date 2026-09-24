"""
Image Quality Enhancement - Backend API

Uses:
- Fine-tuned Real-ESRGAN model for smaller images (<= 350 px)
- Pretrained Real-ESRGAN model for larger images (> 350 px)

Both models run locally on CPU.
"""

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

# NEW: memory-limiting settings
MAX_INPUT_SIDE = 512  # larger uploads are shrunk to fit within this box
TILE = 64             # input tile size; lower = less memory, slower
TILE_PAD = 10         # overlap between tiles to avoid visible seams
SCALE = 4             # Real-ESRGAN x4



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


print("=" * 60)
print("Loading Real-ESRGAN models...")
print("=" * 60)

ft_model = load_model(FT_MODEL_PATH)
pretrained_model = load_model(PRETRAINED_MODEL_PATH)

print("=" * 60)
print("Both models are ready.")
print("=" * 60)



# FASTAPI


app = FastAPI(title="Image Quality Enhancement API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# NEW: TILED INFERENCE

def run_tiled(model, img):
    """Run the model on overlapping tiles so memory use stays small."""

    _, c, h, w = img.shape
    out = torch.zeros((1, c, h * SCALE, w * SCALE))

    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            x1, y1 = min(x0 + TILE, w), min(y0 + TILE, h)

            # Padded bounds (clamped to the image)
            px0, py0 = max(x0 - TILE_PAD, 0), max(y0 - TILE_PAD, 0)
            px1, py1 = min(x1 + TILE_PAD, w), min(y1 + TILE_PAD, h)

            with torch.no_grad():
                t_out = model(img[:, :, py0:py1, px0:px1])

            # Crop the padding back off, in output coordinates
            ox0, oy0 = (x0 - px0) * SCALE, (y0 - py0) * SCALE
            ox1 = ox0 + (x1 - x0) * SCALE
            oy1 = oy0 + (y1 - y0) * SCALE

            out[:, :, y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] = \
                t_out[:, :, oy0:oy1, ox0:ox1]

    return out


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
        model = ft_model
        model_name = "Fine-tuned Real-ESRGAN"
    else:
        model = pretrained_model
        model_name = "Pretrained Real-ESRGAN"

    print(
        f"Enhancing {width}x{height} image "
        f"using {model_name}",
        flush=True
    )

    # PIL -> NumPy
    image_np = np.array(image).astype(np.float32) / 255.0

    # HWC -> CHW
    image_tensor = torch.from_numpy(
        image_np.transpose(2, 0, 1)
    ).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    # CHANGED: run model tile by tile instead of on the whole image at once
    output_tensor = run_tiled(model, image_tensor)

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

    print(f"Output size: {output_image.size}", flush=True)

    return output_image

# API ROUTES

@app.get("/")
def read_root():
    return {
        "status": "Backend is running",
        "message": "Real-ESRGAN enhancement API"
    }


# CHANGED: plain "def" instead of "async def", so the heavy work runs in a
# worker thread and the server can still answer Render's health checks.
@app.post("/enhance")
def enhance(file: UploadFile = File(...)):
    """
    Receive an image, enhance it with Real-ESRGAN,
    and return the enhanced PNG.
    """

    contents = file.file.read()  # CHANGED: was "await file.read()"

    input_image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    # NEW: shrink large uploads (keeps aspect ratio)
    input_image.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE))

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
