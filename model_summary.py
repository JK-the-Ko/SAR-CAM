import argparse

from os import listdir, getcwd, mkdir
from os.path import join

import torch
from torch.onnx import export
from torchsummary import summary

from model import Model

from utils import *

def main() :
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type = str, default = "SAR-CAM")
    parser.add_argument("--input-shape", type = int, default = 64)
    parser.add_argument("--batch-size", type = int, default = 16)
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
                bias = True
                ).to(device)

    # Get Parameters of Current Model
    print(summary(model, (1, args.input_shape, args.input_shape), batch_size = args.batch_size))

    # Initialize Dummy Data for Exporting Model
    dummy_data = torch.empty(1, 1, args.input_shape, args.input_shape, dtype = torch.float32).to(device)

    # Create Directory for Saving ONNX Model
    if "onnx_model" not in listdir(getcwd()) :
        mkdir(join(getcwd(), "onnx_model"))

    # Export Model as ONNX
    export(model, dummy_data, f"onnx_model/{args.model_name}.onnx")

if __name__ == "__main__" :
    main()