%reset -f

import torch
import cv2
import numpy as np
import skfuzzy as fuzz  # For fuzzy logic

# Load the MiDaS model for depth estimation
model_type = "DPT_Large"  # or "DPT_Hybrid" for lighter model
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Load the transform for MiDaS model
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

video_path = 'tst.mp4'  
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set up video writer to save the output enhanced depth video
output_width, output_height = 256, 256  # Decreased dimensions for faster processing
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for output video
output_video = cv2.VideoWriter('Fuzzy_Video_Depth_Map.avi', fourcc, fps, (output_width, output_height), False)

# Define fuzzy membership functions for low, medium, and high depth regions
x_depth = np.linspace(0, 1, 256)  # Depth values between 0 and 1

# Create fuzzy membership functions
depth_low = fuzz.trimf(x_depth, [0, 0, 0.5])  # Low depth
depth_medium = fuzz.trimf(x_depth, [0, 0.5, 1])  # Medium depth
depth_high = fuzz.trimf(x_depth, [0.5, 1, 1])  # High depth

# Process the video frame by frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}/{total_frames}")

    # Pre-process the frame (convert to RGB and resize to smaller input size)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb_resized = cv2.resize(img_rgb, (256, 256))  # Adjust size for faster processing

    # Apply the MiDaS transform (resize and normalize the image)
    input_batch = transform(img_rgb_resized).unsqueeze(0)

    # Ensure input tensor has the correct dimensions
    if len(input_batch.shape) == 5:
        input_batch = input_batch.squeeze(1)

    # Move model and tensor to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    input_batch = input_batch.to(device)

    # Perform inference (depth estimation)
    with torch.no_grad():
        prediction = midas(input_batch)

    # Convert the prediction to a depth map
    depth_map = prediction.squeeze().cpu().numpy()

    # Normalize the depth map for further processing
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)

    # Apply fuzzy logic rules to enhance the depth map
    grad_x = np.gradient(depth_map_normalized, axis=1)
    grad_y = np.gradient(depth_map_normalized, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    depth_adjusted = np.zeros_like(depth_map_normalized)

    for i in range(depth_map_normalized.shape[0]):
        for j in range(depth_map_normalized.shape[1]):
            depth_value = depth_map_normalized[i, j]
            grad_value = gradient_magnitude[i, j]

            # Fuzzify the depth and gradient values
            low_strength = fuzz.interp_membership(x_depth, depth_low, depth_value)
            medium_strength = fuzz.interp_membership(x_depth, depth_medium, depth_value)
            high_strength = fuzz.interp_membership(x_depth, depth_high, depth_value)

            # Rule: if gradient is high and depth is low, increase the depth value
            if grad_value > 0.3:  # A threshold for sharp regions
                depth_adjusted[i, j] = (low_strength * 0.8) + (medium_strength * 0.5) + (high_strength * 0.3)
            else:
                depth_adjusted[i, j] = depth_value  # Keep the depth value as it is for smooth regions

    # Normalize the enhanced depth map for saving
    depth_adjusted = cv2.normalize(depth_adjusted, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Resize the enhanced depth map to match the output video size
    depth_adjusted_resized = cv2.resize(depth_adjusted, (output_width, output_height))

    # Save the enhanced depth map to the video
    output_video.write(depth_adjusted_resized)

cap.release()
output_video.release()

