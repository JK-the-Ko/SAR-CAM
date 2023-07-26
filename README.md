# SAR Image Despeckling Using Continuous Attention Module
### [Paper](https://ieeexplore.ieee.org/document/9633208) | [BibTex](#citation)
## Abstract
Speckle removal process is inevitable in the restoration of synthetic aperture radar (SAR) images. Several variant methods have been proposed for enhancing SAR images over the past decades. However, in recent studies, convolutional neural networks (CNNs) have been widely applied in SAR image despeckling because of their versatility in representation learning. Nonetheless, a fair number of textures of the images are still lost when despeckling using simple CNN structures. To solve this problem, an encoder–decoder architecture was previously proposed. Although this architecture extracts features on different scales and has been shown to yield state-of-the-art performance, it still learns representation locally, resulting in missing overall information of convolutional features. Therefore, we herein introduce a new method for SAR image despeckling (SAR-CAM), which improves the performance of an encoder–decoder CNN architecture by using various attention modules. Moreover, a context block is introduced at the minimum scale to capture multiscale information. The model is trained via a data-driven approach using the gradient descent algorithm with a combination of modified despeckling gain and total variation loss function. Experiments performed on simulated and real SAR data demonstrate that the proposed method achieves significant improvements over state-of-the-art methodologies.

## Introduction
- ### Despeckling Performance
![image](https://user-images.githubusercontent.com/55126482/144549532-fb7c196d-6415-43fc-abde-22ad07e406b6.png)
Results for the freeway image with 1-look speckle noise. (a) Reference. (b) Noisy image. (c) PPB. (d) SAR-BM3D. (e) FANS. (f) SAR-DRN. (g)
HDRANet. (h) U-Net. (i) STD-CNN. (j) MONet. (k) MRDDANet. (l) Proposed Method., respectively.

![image](https://user-images.githubusercontent.com/55126482/144549430-1c1c6545-7c46-456c-b706-9eb69d4bbf09.png)
Results for the parking lot image with 1-look speckle noise. (a) Reference. (b) Noisy image. (c) PPB. (d) SAR-BM3D. (e) FANS. (f) SAR-DRN.
(g) HDRANet. (h) U-Net. (i) STD-CNN. (j) MONet. (k) MRDDANet. (l) Proposed Method., respectively.

## Prerequisites
- Python 3.6
- PyTorch 1.8
- NVIDIA GPU + CUDA cuDNN

## Installation
- ### Clone this repo.
```
git clone https://github.com/JK-the-Ko/SAR-CAM.git
cd SAR-CAM/
```
- ### Install PyTorch and dependencies from http://pytorch.org
- ### Please install dependencies by
```
python3 -m pip install -r requirements.txt
```

## Usage
- ### Train
```
python3 train.py --project PROJECT_NAME --noisy-train-dir NOISY_IMAGE_TRAIN_DIR --clean-train-dir CLEAN_IMAGE_TRAIN_DIR --noisy-valid-dir NOISY_IMAGE_VALID_DIR --clean-valid-dir CLEAN_IMAGE_VALID_DIR 
```
- ### Inference
```
python3 test.py --weights-dir SAVE_WEIGHT_DIR --clean-image-dir CLEAN_IMAGE_TEST_DIR --noisy-image-dir NOISY_IMAGE_TEST_DIR --save-dir DENOISED_IMAGE_SAVE_DIR
```

## Plan
- ### Add pretrained models.

## Citation
If you use SAR-CAM in your work, please consider citing us as

```
@ARTICLE{9633208,
  author={Ko, Jaekyun and Lee, Sanghwan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={SAR Image Despeckling Using Continuous Attention Module}, 
  year={2022},
  volume={15},
  number={},
  pages={3-19},
  doi={10.1109/JSTARS.2021.3132027}}
```
