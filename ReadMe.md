# CSCI 490 Capstone

Austin Phanachone

---

# How to Run/Test

I would recommend running the notebook on Google colab. It should take no more than 20 minutes to complete training.
After this, the last cell will continuously run and prompt you to upload an image.
There is a provided folder of test sign language images of the particularly diffucult signs to identify.
The current model will misidentify the signs or have a low confidence percentage.

You can also upload your own images to see how it runs.

The MediaPipe Implementation can be ignored.

[Roboflow Dataset](https://app.roboflow.com/aphanachone/sign-language-project-colorful-d0wkv/)

---

## 1. Project Title & Purpose

# Sign Language Detection with YOLOv11

This project trains a YOLOv11 object detection model to recognize and classify American Sign Language (ASL) hand signs for the numbers 1 - 10 and letters A-Z. The model is trained on a labeled image dataset sourced from Roboflow and evaluated using standard detection metrics — mAP, precision, and recall.

The goal is to demonstrate that a lightweight YOLO model can achieve reliable ASL sign classification. I ultimately hope to run this model locally and utilize a webcam to feed live footage to the model and have it recognize signs in real time.

---

## 2. Key Features Implemented

- Dataset of 3600 images from Roboflow for use in YOLOv11
- YOLOv11 nano (yolo11n.pt) fine-tuned from COCO pretrained weights across 20 epochs
- Validation on the val split and a separate held-out test set evaluation with plots enabled
- Post-training evaluation showing training loss and mAP curves, a confusion matrix from the test set, and a precision-recall curve
-]Upload cell allowing any image to be tested against the trained model, with a sorted confidence breakdown per detection

---

## 3. Languages, Libraries & Frameworks

- **Python 3.12** — primary language for all notebook cells
- **ultralytics 8.3** — YOLOv11 model architecture, training loop, and evaluation
- **roboflow** — dataset versioning and download client
- **PyTorch (CUDA)** — deep learning backend used by ultralytics
- **pandas** — reads results.csv to build training curve plots
- **matplotlib** — renders all training curve and metric figures
- **seaborn** — annotated confusion matrix heatmap
- **NumPy** — array operations for confusion matrix extraction
- **PyYAML** — parses data.yaml to load class names
- **Google Colab (T4 GPU)** — primary training environment
- **Roboflow Universe (Dataset v3)** — ASL sign language labeled image dataset

---

## 4. Challenges Faced & How They Were Addressed

### Model Generalization Across Similar and Dynamic Signs

ASL contains several signs that are visually similar in still-image form. Letters such as N and M share nearly identical hand shapes with only minor finger positioning differences. Training a detection model to reliably distinguish these classes is difficult  when the dataset contains limited variation in hand view angle and skin tone. In addition, some signs, like 10, J, and Z, are dynamic and have a moving component.

While these issues haven’t been properly addressed, I am currently working on analysis of the database from Roboflow to see where additional metrics can be applied to better quantify per-class confusion. I am currently looking to add to the database with ASL hand signs from differing angles and lighting to help lower class confusion. Additional epochs in training with the code could also give the model more time to learn subtle shape differences.

For dynamic signs, there isn’t a definitive method to identify the signs and their associated motions with YOLOv11, but they have unique positions near the start of their motion which can be trained as their sign.

### Training Speed and Hardware Constraints

Each training epoch is computationally expensive when running on CPU or a low-VRAM GPU, making iteration slow during development. On a local machine without CUDA properly configured, PyTorch uses thel device’s CPU, which can make epochs take many times longer without any visible error

I moved training to Google Colab with its T4 GPU to take advantage of available VRAM and CUDA support. Mixed-precision training via `amp=True` and image caching via `cache=True` were identified as the primary parameters to reduce per-epoch time further.

I aim to move to my local machine once I configure the notebook properly to my device.

---

## 5. Future Improvement Areas

- Train for more epochs (50+) with early stopping (`patience=10`) to allow the model to converge more fully without overfitting as seen in the training graphs
- Expand the dataset with additional augmentations for lighting variation and hand orientation
- Integrate webcam inference using OpenCV for live sign detection, making the model usable as a practical accessibility tool
- Get model running on local device
