import os
import cv2
from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "models\signboard_model.onnx"
yolov8_detector = YOLOv8(model_path)

# Local path to the input video file
input_video_path = "video_1.mp4"

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

frame_counter = 0  # Counter to keep track of frames

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    frame_counter += 1  # Increment frame counter

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

        print(f"Box {i+1}: x={x}, y={y}, w={w}, h={h}")

        if 0 <= x < frame_width and 0 <= y < frame_height and 0 < w <= frame_width and 0 < h <= frame_height:
            cropped_img = frame[y:h, x:w]
            save_path = os.path.join(output_dir, f"frame_{frame_counter}_box_{i+1}.jpg")
            cv2.imwrite(save_path, cropped_img)
        else:
            print(f"Invalid box coordinates for Box {i+1}")

    # Show the processed frame with detections
    cv2.imshow("Detected Objects", combined_img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()
