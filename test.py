import argparse

from os import listdir
from os.path import join

import PIL.Image as pil_image
import PIL.ImageFilter as pil_image_filter

import cv2
import numpy as np
import pandas as pd

import torch
from torchvision import transforms

from model import Model
from utils import calc_psnr, calc_ssim, set_logging, select_device

from tqdm import tqdm

def main() :
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type = str, default = "SAR-CAM")
    parser.add_argument("--weights-dir", type = str, required = True)
    parser.add_argument("--clean-image-dir", type = str, required = True)
    parser.add_argument("--noisy-image-dir", type = str, required = True)
    parser.add_argument("--save-dir", type = str, required = True)
    parser.add_argument("--stack-image", action = "store_true")
    parser.add_argument("--device", default = "", help = "cuda device, i.e. 0 or 0,1,2,3 or cpu")
    args = parser.parse_args()

    # Get Current Namespace
    print(args)

    # Assign Device
    set_logging()
    device = select_device(args.model_name, args.device)

    # Create Model Instance
    model = Model(
                scale = 2,
                in_channels = 1,
                channels = 128,
                kernel_size = 3,
                stride = 1,
                dilation = 1,
                bias = False
                ).to(device)
    model.load_state_dict(torch.load(args.weights_dir))

    # Create Torchvision Transforms Instance
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Create List Instance for Saving Metrics
    image_name_list, psnr_noisy_list, psnr_denoised_list, ssim_noisy_list, ssim_denoised_list = list(), list(), list(), list(), list()

    # Assign Device
    model.to(device)

    # Evaluate Model
    model.eval()

    with tqdm(total = len(listdir(args.noisy_image_dir))) as pbar :
        with torch.no_grad() :
            for x in listdir(args.noisy_image_dir) :
                # Get Image Path
                clean_image_path = join(args.clean_image_dir, x)
                noisy_image_path = join(args.noisy_image_dir, x)

                # Load Image
                clean_image = pil_image.open(clean_image_path)
                noisy_image = pil_image.open(noisy_image_path)

                # Convert Pillow Image to PyTorch Tensor
                tensor_clean_image = to_tensor(clean_image).unsqueeze(0)
                tensor_noisy_image = to_tensor(noisy_image).unsqueeze(0).to(device)

                # Get Prediction
                pred = model(tensor_noisy_image)

                # Assign Device into CPU
                tensor_noisy_image = tensor_noisy_image.detach().cpu()
                pred = pred.detach().cpu()

                # Calculate PSNR
                psnr_noisy = calc_psnr(tensor_noisy_image, tensor_clean_image).item()
                psnr_denoised = calc_psnr(pred, tensor_clean_image).item()

                # Calculate SSIM
                ssim_noisy = calc_ssim(tensor_noisy_image, tensor_clean_image,size_average = True).item()
                ssim_denoised = calc_ssim(pred, tensor_clean_image, size_average = True).item()

                # Append Image Name
                image_name_list.append(x)

                # Append PSNR
                psnr_noisy_list.append(psnr_noisy)
                psnr_denoised_list.append(psnr_denoised)

                # Append SSIM
                ssim_noisy_list.append(ssim_noisy)
                ssim_denoised_list.append(ssim_denoised)

                # Convert PyTorch Tensor to Pillow Image
                pred = torch.clamp(pred, min = 0.0, max = 1.0)
                pred = to_pil(pred.squeeze(0))

                if args.stack_image :
                    # Get Edge
                    noisy_image_edge = noisy_image.filter(pil_image_filter.FIND_EDGES)
                    pred_edge = pred.filter(pil_image_filter.FIND_EDGES)
                    clean_image_edge = clean_image.filter(pil_image_filter.FIND_EDGES)

                    # Convert into Numpy Array
                    noisy_image = np.array(noisy_image, dtype = "uint8")
                    pred = np.array(pred, dtype = "uint8")
                    clean_image = np.array(clean_image, dtype = "uint8")

                    noisy_image_edge = np.array(noisy_image_edge, dtype = "uint8")
                    pred_edge = np.array(pred_edge, dtype = "uint8")
                    clean_image_edge = np.array(clean_image_edge, dtype = "uint8")

                    # Stack Images
                    stacked_image_clean = np.hstack((noisy_image, pred, clean_image))
                    stacked_image_edge = np.hstack((noisy_image_edge, pred_edge, clean_image_edge))
                    stacked_image = np.vstack((stacked_image_clean, stacked_image_edge))

                    # Save Image
                    cv2.imwrite(f"{args.save_dir}/{x}", stacked_image)

                else :
                    # Save Image
                    pred.save(f"{args.save_dir}/{x}")

                # Update TQDM Bar
                pbar.update()

    # Create Dictionary Instance
    d = {"Noisy Image PSNR(dB)" : psnr_noisy_list,
            "Noisy Image SSIM" : ssim_noisy_list,
            "Denoised Image PSNR(dB)" : psnr_denoised_list,
            "Denoised Image SSIM" : ssim_denoised_list}

    # Create Pandas Dataframe Instance
    df = pd.DataFrame(data = d, index = image_name_list)

    # Save as CSV Format
    df.to_csv(f"{args.save_dir}/image_quality_assessment.csv")

if __name__ == "__main__" :
    main()