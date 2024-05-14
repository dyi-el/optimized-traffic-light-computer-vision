# Optimized Traffic Light with YOLOv8
![Static Badge](https://img.shields.io/badge/status-Work_in_Progress-blue)

Designing an Optimized Traffic Light Control System using a hybrid of Image Classification and Object Detection
## Overview 


This is the codebase of a college academic capstone research *(on going)*. It uses YOLOv8 to train the models to be used for inferencing. It has proven itself good for fast and reliable inference.
[See Ultralytics](https://github.com/ultralytics/ultralytics)
<img src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png" alt="block"/>
It is also good for use to both image classification and object detection.
## Requirements

To install the dependencies with pipenv:
*Assuming you have pipenv already installed*

```bash
pipenv install
```

## Instructions
1. Install requirements through the Pipfile.
2. Use desired microprocessor. This project uses Beaglebone Black and Arduino. Install the desired packages to control GPIO pins. For Arduino, StandardPyFirmata was installed.
3. Copy and run traffic-light.py on microprocessor and connect pins with the arduino. Be sure to keep check HIGH voltage levels of devices. (Arduino=5V, RPi,BBB=3.3V)
4. Activate the virtual shell environment of pipenv.
```bash
pipenv shell
```
5. Run actuator.py on shell. Using your computer.



## Connections Diagram

<img src="/assets/block.png" alt="block"/>


(in progress)
