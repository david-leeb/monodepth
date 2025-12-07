import os
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import networks

def run_inference(image_path, weights_folder):
    # 1. Setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"-> Loading model from {weights_folder}")

    # 2. Load Architecture (ResNet18 + DepthDecoder default)
    encoder_path = os.path.join(weights_folder, "encoder.pth")
    decoder_path = os.path.join(weights_folder, "depth.pth")
    
    encoder_dict = torch.load(encoder_path)
    height = encoder_dict['height']
    width = encoder_dict['width']
    
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    # 3. Load Weights
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.to(device)
    depth_decoder.to(device)
    encoder.eval()
    depth_decoder.eval()

    # 4. Load and Preprocess Image
    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((width, height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    input_image = input_image.to(device)

    # 5. Inference
    print("-> Computing depth...")
    with torch.no_grad():
        features = encoder(input_image)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]

    # 6. Save Result
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)
    
    # Save as a colorful heatmap
    name_dest_im = os.path.splitext(os.path.basename(image_path))[0] + "_depth.jpeg"
    plt.imsave(name_dest_im, disp_resized.squeeze().cpu().numpy(), cmap='magma', format='jpeg')
    print(f"-> Saved prediction to {name_dest_im}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--load_weights_folder", type=str, required=True)
    args = parser.parse_args()

    run_inference(args.image_path, args.load_weights_folder)
