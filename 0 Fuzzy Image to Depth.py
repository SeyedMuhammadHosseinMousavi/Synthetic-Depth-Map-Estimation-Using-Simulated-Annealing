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

# Load the input image
image_path = 'img2.jpg'  
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize the image and apply transformation
img_rgb_resized = cv2.resize(img_rgb, (256, 256))  # Adjust size as needed
input_batch = transform(img_rgb_resized).unsqueeze(0)  # Add batch dimension

if len(input_batch.shape) == 5:  # Remove any extra dimension if present
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

# ---- Apply Fuzzy Logic to Enhance Depth Map ---- #
# Define fuzzy membership functions for low, medium, and high depth regions
x_depth = np.linspace(0, 1, 256)  # Depth values between 0 and 1

# Create fuzzy membership functions
depth_low = fuzz.trimf(x_depth, [0, 0, 0.5])  # Low depth
depth_medium = fuzz.trimf(x_depth, [0, 0.5, 1])  # Medium depth
depth_high = fuzz.trimf(x_depth, [0.5, 1, 1])  # High depth

# Apply fuzzy rules based on gradient of the depth map
grad_x = np.gradient(depth_map_normalized, axis=1)
grad_y = np.gradient(depth_map_normalized, axis=0)
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Define fuzzy rules: if the gradient is high, then we adjust the depth value
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

# Normalize and save the adjusted depth map
depth_adjusted = cv2.normalize(depth_adjusted, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('Fuzzy Image_Depth_Map.png', depth_adjusted)

