# SAR Image Despeckling Using Continuous Attention Module

## Model Architecture
![image](https://user-images.githubusercontent.com/55126482/144569524-0bd02952-cbef-4413-af47-398ebed6441c.png)

## Despeckling Performance
![image](https://user-images.githubusercontent.com/55126482/144549532-fb7c196d-6415-43fc-abde-22ad07e406b6.png)
Results for the freeway image with 1-look speckle noise. (a) Reference. (b) Noisy image. (c) PPB. (d) SAR-BM3D. (e) FANS. (f) SAR-DRN. (g)
HDRANet. (h) U-Net. (i) STD-CNN. (j) MONet. (k) MRDDANet. (l) Proposed Method., respectively.

![image](https://user-images.githubusercontent.com/55126482/144549430-1c1c6545-7c46-456c-b706-9eb69d4bbf09.png)
Results for the parking lot image with 1-look speckle noise. (a) Reference. (b) Noisy image. (c) PPB. (d) SAR-BM3D. (e) FANS. (f) SAR-DRN.
(g) HDRANet. (h) U-Net. (i) STD-CNN. (j) MONet. (k) MRDDANet. (l) Proposed Method., respectively.

## Usage
```
python3 train.py --project PROJECT_NAME --noisy-train-dir NOISY_IMAGE_TRAIN_DIR --clean-train-dir CLEAN_IMAGE_TRAIN_DIR --noisy-valid-dir NOISY_IMAGE_VALID_DIR --clean-valid-dir CLEAN_IMAGE_VALID_DIR 
```

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
