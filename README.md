Signboard_Detection_And_Recognition
====================================
Signboard_Detection_And_Recognition using fine-tuned YOLO V8 model with ImageNet weights trained on Roboflow custom dataset (Runs CPU Inference Using ONNX) and text extraction using PaddleOCR.

<center><img src="readme_media\output_video1-ezgif.gif" alt="your alt text" width="300"/></center>

## Introduction:

- The repository includes a training notebook that demonstrates how to train the model. The notebook provides step-by-step instructions and includes the output of each step for reference.
- The trained Signboard model was exported to the ONNX format. A Python script is provided in the repository that demonstrates how to perform CPU inference using the ONNX model with the 'onnxruntime' library. This allows for fast and efficient inference of object detection and OCR on CPU-based systems.
- The script takes in a sample video and detects signboards in each frame. 
- The detected frames are passed through OCR using PaddleOCR pretrained model and the output is added to an excel sheet along with the timestamp of the signboard occurence in the sample video.

## Prerequisites
* [Python 3.10.11](https://www.python.org/downloads/release/python-31011/)
* Sample video link : [https://www.youtube.com/watch?v=_pHZqBQYRQ4](https://www.youtube.com/watch?v=_pHZqBQYRQ4)
* Roboflow Dataset : [Signboard Detection Dataset by HCMUT](https://universe.roboflow.com/hcmut-ek6t5/signboard-detection-svdwo/dataset/2/images)

## Installation

1. Install virtualenv

    ```
    $ pip install virtualenv
    ```    

2. Create a virtualenv named 'signboard_env'

    ```
   $  virtualenv -p python3.10 signboard_env
    ```

3. Activate the environment

    ```
    $ source iris_env/bin/activate
    ```
4. Install the requirements

    ```
    $ pip install -r requirements.txt
    ```

5. Run onnx_signboard_ocr_xlsx_v2.py

    ```
    $ python onnx_signboard_ocr_xlsx_v2.py
    ```

<br>

# Results
<br>


<img src="readme_media\results.png" alt="your alt text" width="600"/>

<br>

<img src="readme_media\val_batch0_pred.jpg" alt="your alt text" width="600"/>

<img src="readme_media\output_video2-ezgif.gif" alt="your alt text" width="600"/>



