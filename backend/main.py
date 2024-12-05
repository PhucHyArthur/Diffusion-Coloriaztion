from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import logging
from load import colordiff_model  # Load the pretrained model
from torchvision import transforms

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Initialize FastAPI app
app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


import base64
import cv2
import numpy as np
import torch

def preprocess_image_base64(base64_string: str) -> torch.Tensor:
    """
    Decode a base64 image string and preprocess it for the model.
    Ensure the output tensor has torch.Size([1, 1, 128, 128]).
    """
    # Remove MIME information if present
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]

    # Fix incorrect padding
    missing_padding = len(base64_string) % 4
    if missing_padding != 0:
        base64_string += '=' * (4 - missing_padding)

    try:
        # Decode base64 string
        img_data = base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Base64 decoding failed: {str(e)}")

    # Convert to NumPy array and decode using OpenCV
    nparr = np.frombuffer(img_data, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Image decoding failed. Invalid Base64 string or unsupported image format.")

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_lab = ((img_lab * 2.0) / 255.0) - 1.0

    # Add channel and batch dimensions
    tens_img_lab = torch.tensor(img_lab.transpose(2, 0, 1), dtype=torch.float32)
    original_img_l = tens_img_lab[:1, :, :]
    resize_transform = transforms.Resize((128, 128))
    tens_img_lab = resize_transform(tens_img_lab.unsqueeze(0)).squeeze(0)
    tens_img_l = tens_img_lab[:1, :, :]
    tens_img_ab = tens_img_lab[1:, :, :]
    return original_img_l, tens_img_l, tens_img_ab



def inference(model, img_tensor: torch.Tensor):
    """
    Perform inference using the PyTorch model.
    """
    with torch.no_grad():
        output, visuals = model.restoration(img_tensor[1].unsqueeze(0).cpu())
    return output, visuals


def postprocess_output(img_l: torch.Tensor, img_ab: torch.Tensor) -> BytesIO:
    """
    Convert the L and ab tensors to an RGB image buffer for streaming response.
    """
   # Shape: (H, W, 1)
    img_l = img_l.permute(1, 2, 0).cpu().numpy()
    img_ab = img_ab.permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, 2)

    # Resize ab channels to match the dimensions of L
    img_ab = cv2.resize(img_ab, (img_l.shape[1], img_l.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Concatenate L and ab channels to form LAB image
    arr_lab = np.concatenate([img_l, img_ab], axis=2)  # Shape: (H, W, 3)

    # Denormalize LAB image values from [-1, 1] to [0, 255]
    arr_lab = (arr_lab + 1.0) * 255 / 2
    arr_lab = np.clip(arr_lab, 0, 255).astype(np.uint8)
    # Ensure the LAB image has exactly 3 channels
    # if arr_lab.shape[2] != 3:
    #     raise ValueError(f"LAB image must have 3 channels, but got {arr_lab.shape[2]} channels.")

    # Convert LAB to BGR
    arr_bgr = cv2.cvtColor(arr_lab, cv2.COLOR_LAB2BGR)

    # Convert BGR to RGB
    arr_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)

    # Convert NumPy array to PIL Image
    image = Image.fromarray(arr_rgb)

    # Save image to buffer
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)  # Reset buffer pointer to the beginning

    return img_buffer


@app.post("/transfer")
async def transfer_image(payload: dict):
    """
    Endpoint to process uploaded image (base64) and return colorized image.
    """
    try:
        # Extract base64 image from payload
        base64_image = payload.get("image")
        if not base64_image:
            raise HTTPException(status_code=400, detail="No base64 image found in the payload.")

        # Preprocess the image
        img_tensor = preprocess_image_base64(base64_image)

        # Perform inference
        output, visuals = inference(colordiff_model, img_tensor)
        logging.debug("Inference completed successfully.")

        # Postprocess and return the result
        img_buffer = postprocess_output(img_tensor[0], output[0])
        return StreamingResponse(img_buffer, media_type="image/png")

    except ValueError as ve:
        logging.error(f"ValueError: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        logging.exception("Error during image processing.")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
