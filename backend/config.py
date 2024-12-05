# config.py

unet_config = {
    "in_channel": 3,
    "out_channel": 2,
    "inner_channel": 64,
    "channel_mults": [1, 2, 4, 8],
    "attn_res": [16],
    "num_head_channels": 32,
    "res_blocks": 2,
    "dropout": 0.2,
    "image_size": 128
}

beta_schedule = {
    "schedule": "linear",
    "n_timestep": 20,
    "linear_start": 1e-4,
    "linear_end": 0.09
}
