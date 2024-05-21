# Drowsiness Detection System

## Overview

The Drowsiness Detection System is designed to alert drivers when they become drowsy, helping to prevent accidents caused by fatigue. This system uses a webcam to monitor the driver's eye aspect ratio (EAR) and triggers an alert when the eyes remain closed for a certain number of consecutive frames. The alert includes both an audible alarm and visual feedback.

## Features

- Real-time eye aspect ratio calculation
- Audible alarm using Pygame
- Visual feedback with blinking text
- Adjustable thresholds for EAR and consecutive frames

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.x
- OpenCV
- dlib
- imutils
- numpy
- scipy
- pygame

You can install the required packages using pip:

```bash
pip install opencv-python dlib imutils numpy scipy pygame
