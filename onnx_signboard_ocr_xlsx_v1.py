import os
import cv2
import concurrent.futures
import threading
from yolov8 import YOLOv8
from paddleocr import PPStructure, save_structure_res
import pandas as pd
from datetime import datetime

# Initialize yolov8 object detector
model_path_yolo = "models\signboard_model.onnx"
yolov8_detector = YOLOv8(model_path_yolo)

# Initialize PaddleOCR
table_engine = PPStructure(recovery=True, lang='en', show_log=True)

# Create an empty DataFrame
paddleocr_df = pd.DataFrame(columns=["shop_names", "timestamp"])

# Function to run PaddleOCR on an image and store the output in the DataFrame
def run_paddleocr_and_store(img_path, output_folder, frame_counter):
    # Load the image using OpenCV
    img = cv2.imread(img_path)

    # Check if the image was loaded successfully
    if img is not None:
        # Call the OCR function with the image
        result = table_engine(img, img_idx=frame_counter)

        # Access the "res" field from the existing result variable
        res_structure = result[0].get("res", [])

        # Extract text values into a list
        final_info = [entry.get("text", "") for entry in res_structure]

        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Print or do whatever you want with the final information
        print(f"Frame {frame_counter} OCR Information:", final_info)

        # Append the PaddleOCR output and timestamp to the DataFrame
        global paddleocr_df
        paddleocr_df = pd.concat([paddleocr_df, pd.DataFrame({"shop_names": [final_info], "timestamp": [timestamp]})],
                                 ignore_index=True)

# Local path to the input video file
input_video_path = "sample_videos/video_cut1.mp4"

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get video details
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
output_video_path = "output_videos/output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Change codec here
out = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 640))

# Define the directory to save cropped images
output_dir = "cropped_regions"

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a lock for thread-safe access to OpenCV window
cv2_lock = threading.Lock()

# Create the OpenCV window
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# Process frames without using ThreadPoolExecutor
frame_counter = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_counter += 1

    # Resize the frame to 640x640
    frame_resized = cv2.resize(frame, (640, 640))

    # Detect Objects
    boxes, scores, class_ids = yolov8_detector(frame_resized)

    # Draw detections
    combined_img = yolov8_detector.draw_detections(frame_resized)

    # Save the frame with detections to the output video
    out.write(combined_img)

    # Crop and save bounding box region
    for i, box in enumerate(boxes):
        x, y, w, h = map(int, box)  # Convert to integers

        if 0 <= x < frame_width and 0 <= y < frame_height and 0 < w <= frame_width and 0 < h <= frame_height:
            cropped_img = frame[y:h, x:w]
            save_path = os.path.join(output_dir, f"frame_{frame_counter}_box_{i+1}.jpg")
            cv2.imwrite(save_path, cropped_img)

            # Run PaddleOCR on the cropped image and store the result in DataFrame
            run_paddleocr_and_store(save_path, output_dir, frame_counter)

    # Show the processed frame with detections
    with cv2_lock:
        cv2.imshow("Detected Objects", combined_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the DataFrame to an Excel file
paddleocr_df.to_excel("paddleocr_output.xlsx", index=False)

# Release the video capture and writer objects
cap.release()
out.release()

# Close the OpenCV window gracefully
cv2.destroyAllWindows()
