"""
Image Quality Enhancement - Backend API

Uses:
- Pretrained Real-ESRGAN model
- Two enhancement passes
- Maximum output size of 2048 px per side
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

# Final maximum image dimension.
# This prevents repeated 4x upscaling from creating
# excessively large images.
MAX_OUTPUT_SIDE = 2048

# Mentor-selected strategy:
# Pretrained Real-ESRGAN applied twice.
NUM_PASSES = 2

PRETRAINED_MODEL_PATH = MODELS_PATH / "RealESRGAN_x4plus.pth"



# MODEL LOADING


def create_model():
    """Create the RRDBNet architecture used by Real-ESRGAN."""

    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )


def load_model(model_path: Path):
    """Load the pretrained Real-ESRGAN model."""

    print(f"Loading model: {model_path.name}")

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    # The pretrained Real-ESRGAN checkpoint uses "params_ema".
    if "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif "params" in checkpoint:
        state_dict = checkpoint["params"]
    else:
        state_dict = checkpoint

    model = create_model()

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print(f"Loaded successfully: {model_path.name}")

    return model


print("=" * 60)
print("Loading Real-ESRGAN model...")
print("=" * 60)

pretrained_model = load_model(
    PRETRAINED_MODEL_PATH
)

print("=" * 60)
print("Pretrained Real-ESRGAN is ready.")
print("=" * 60)



# FASTAPI


app = FastAPI(
    title="Image Quality Enhancement API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# HELPER FUNCTIONS


def resize_for_safe_output(
    image: Image.Image,
    max_output_side: int = MAX_OUTPUT_SIDE
):
    """
    Resize an image before a 4x Real-ESRGAN pass if the
    predicted output would exceed the maximum allowed size.

    Returns:
        resized_image
        whether a resize occurred
    """

    width, height = image.size

    predicted_width = width * 4
    predicted_height = height * 4

    predicted_max_side = max(
        predicted_width,
        predicted_height
    )

    # No resize needed
    if predicted_max_side <= max_output_side:
        return image, False

    # Calculate the scale needed to keep the output safe
    scale = max_output_side / predicted_max_side

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    resized_image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    return resized_image, True


def run_one_pass(
    image: Image.Image
) -> Image.Image:
    """
    Run one pretrained Real-ESRGAN enhancement pass.
    """

    # PIL -> NumPy
    image_np = (
        np.array(image)
        .astype(np.float32)
        / 255.0
    )

    # HWC -> CHW
    image_tensor = torch.from_numpy(
        image_np.transpose(2, 0, 1)
    ).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    # Run model
    with torch.no_grad():
        output_tensor = pretrained_model(
            image_tensor
        )

    # Tensor -> NumPy
    output_np = (
        output_tensor
        .squeeze(0)
        .cpu()
        .clamp(0, 1)
        .numpy()
    )

    # CHW -> HWC
    output_np = output_np.transpose(1, 2, 0)

    # Float -> uint8
    output_np = (
        output_np * 255.0
    ).round().astype(np.uint8)

    # NumPy -> PIL
    output_image = Image.fromarray(
        output_np
    )

    return output_image



# IMAGE ENHANCEMENT


def enhance_image(
    image: Image.Image
) -> Image.Image:
    """
    Enhance an image using the pretrained Real-ESRGAN
    model for two consecutive passes.

    Each pass performs 4x upscaling.

    A safety resize is applied before a pass if the
    predicted output would exceed 2048 px on either side.
    """

    current_image = image.copy()

    print("=" * 60)
    print(
        f"Enhancing image: "
        f"{current_image.size[0]}x"
        f"{current_image.size[1]}"
    )
    print(
        "Strategy: Pretrained Real-ESRGAN × 2 passes"
    )
    print("=" * 60)

    for pass_number in range(1, NUM_PASSES + 1):

        original_size = current_image.size

        # Check whether the next 4x pass would
        # create an excessively large image.
        safe_image, was_resized = resize_for_safe_output(
            current_image
        )

        if was_resized:

            print(
                f"Pass {pass_number}/{NUM_PASSES} | "
                f"Safety resize: "
                f"{original_size} → "
                f"{safe_image.size}"
            )

            current_image = safe_image

        input_width, input_height = current_image.size

        predicted_width = input_width * 4
        predicted_height = input_height * 4

        print(
            f"Pass {pass_number}/{NUM_PASSES} | "
            f"Pretrained Real-ESRGAN | "
            f"{current_image.size} → "
            f"{predicted_width}x{predicted_height}"
        )

        # Run Real-ESRGAN
        current_image = run_one_pass(
            current_image
        )

        print(
            f"Pass {pass_number} complete | "
            f"Output: {current_image.size}"
        )

    print("=" * 60)
    print(
        f"Final output size: "
        f"{current_image.size}"
    )
    print("=" * 60)

    return current_image


# API ROUTES

@app.get("/")
def read_root():

    return {
        "status": "Backend is running",
        "message": (
            "Pretrained Real-ESRGAN "
            "enhancement API"
        ),
        "strategy": "Pretrained model - 2 passes"
    }


@app.post("/enhance")
async def enhance(
    file: UploadFile = File(...)
):
    """
    Receive an image, enhance it using the
    pretrained Real-ESRGAN model twice,
    and return the enhanced PNG.
    """

    contents = await file.read()

    input_image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    output_image = enhance_image(
        input_image
    )

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
