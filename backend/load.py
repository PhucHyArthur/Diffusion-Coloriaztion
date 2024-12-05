import torch
from config import unet_config, beta_schedule
from diffusion import ColorDiffusion

# Instantiate the model
colordiff_model = ColorDiffusion(unet_config, beta_schedule)

# Specify the device (e.g., 'cpu' or 'cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pass the device to the function
colordiff_model.set_new_noise_schedule(device)

# Load the model state
load_state = torch.load('D:/AI_OpenSource/Model/model_stage2.pth', map_location=device, weights_only=True)

# Load the state dictionary into the model
colordiff_model.load_state_dict(load_state, strict=True)

# Set the model to evaluation mode and move it to the device
colordiff_model.eval()
colordiff_model = colordiff_model.to(device)  # Correctly move the model to the device
