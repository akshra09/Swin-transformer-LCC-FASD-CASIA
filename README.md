# Swin-transformer-LCC-FASD-CASIA

This repository contains the implementation of a deepfake detection model using the Swin Transformer architecture. The model is trained and evaluated on the Large Crowdcollected Facial Anti-Spoofing Dataset (LCC-FASD), CASIA dataset, and the combined dataset of LCC-FASD-CASIA.

Introduction
Deepfake detection is an important task in multimedia forensics, aimed at identifying manipulated or synthetic content, particularly in images and videos. The Swin Transformer architecture has shown promising results in various computer vision tasks, including image classification and object detection. In this project, we leverage the power of Swin Transformer to develop a robust deepfake detection model.

Dataset
Large Crowdcollected Facial Anti-Spoofing Dataset (LCC-FASD)
Training: 1302 real images and 7444 spoof images
Validation: 416 real images and 2590 spoof images
Evaluation: 323 real images and 7312 spoof images
CASIA Dataset
The CASIA dataset contains a collection of real and fake images, with details such as image resolution, color depth, and compression format.

The dataset can be found here:
https://www.kaggle.com/datasets/ahmedruhshan/lcc-fasd-casia-combined

Usage

Clone the repository:
bash
git clone https://github.com/your-username/Swin-transformer-LCC-FASD-CASIA.git
cd Swin-transformer-LCC-FASD-CASIA

Results
The trained Swin Transformer model achieved an accuracy of 75% on real images and 96% on spoof images, demonstrating its effectiveness in deepfake detection.

Acknowledgments
We would like to thank the authors of the Swin Transformer paper for their contribution to the field of computer vision.

Results
The trained Swin Transformer model achieved an accuracy of 80% on real images and 96% on spoof images, demonstrating its effectiveness in deepfake detection.

Acknowledgments
We would like to thank the authors of the Swin Transformer paper for their contribution to the field of computer vision.
