%reset -f
import torch
import cv2
import numpy as np
import random
import math

# Load the MiDaS model
model_type = "DPT_Large"  # "DPT_Hybrid" for a lighter model
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Load the transform for the model
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Load image
image_path = 'img2.jpg'  
img = cv2.imread(image_path)

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# Pre-process the image (convert to RGB and resize to smaller input size)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb_resized = cv2.resize(img_rgb, (256, 256)) 
# Apply the MiDaS transform (resize and normalize the image)
input_batch = transform(img_rgb_resized).unsqueeze(0)  # Unsqueeze to add batch dimension

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

# Objective function for simulated annealing with edge preservation
def objective_function(depth_map, edge_map):
    """Objective function to minimize sharpness while preserving edges."""
    grad_x = np.gradient(depth_map, axis=1)
    grad_y = np.gradient(depth_map, axis=0)
    smoothness = (1 - edge_map) * (np.var(grad_x) + np.var(grad_y))  # Prioritize non-edge areas
    return np.var(smoothness)

# Detect edges in the depth map to preserve
def detect_edges(depth_map):
    """Detect edges in the depth map using Sobel filter."""
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.magnitude(grad_x, grad_y)
    return (edges > edges.mean()).astype(float)  # Binary edge map

# Outlier removal by clamping extreme depth values
def remove_outliers(depth_map, low_thresh=5, high_thresh=95):
    """Clamp depth map to remove extreme outliers."""
    return np.clip(depth_map, np.percentile(depth_map, low_thresh), np.percentile(depth_map, high_thresh))

# Generate a neighboring solution by randomly adjusting a small portion of the depth map
def neighbor(depth_map):
    """Generate a neighboring solution by randomly modifying a small portion of the depth map."""
    new_map = depth_map.copy()
    i, j = random.randint(0, depth_map.shape[0]-1), random.randint(0, depth_map.shape[1]-1)
    new_map[i, j] = random.uniform(0, 1)  # Randomly adjust one pixel
    return new_map

# simulated annealing 
def simulated_annealing(depth_map, edge_map, objective_function, T=1.0, T_min=0.0001, alpha=0.9, max_neighbors=30):
    """Apply simulated annealing to optimize the depth map with edge preservation."""
    current_map = depth_map
    best_map = depth_map
    current_score = objective_function(current_map, edge_map)
    best_score = current_score

    while T > T_min:
        for _ in range(max_neighbors):  # Fewer neighbors for reduced complexity
            new_map = neighbor(current_map)
            new_score = objective_function(new_map, edge_map)
            delta_score = new_score - current_score

            # If the new score is better, accept it
            if delta_score < 0 or random.uniform(0, 1) < math.exp(-delta_score / T):
                current_map = new_map
                current_score = new_score

                # Update the best solution
                if current_score < best_score:
                    best_map = current_map
                    best_score = current_score

        # Decrease the temperature
        T *= alpha

    return best_map

# Detect edges in the depth map to preserve them
edge_map = detect_edges(depth_map)

# Remove outliers from the depth map
depth_map = remove_outliers(depth_map)

# Normalize the depth map for further processing
depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)

# Apply simulated annealing to optimize the depth map
optimized_depth_map = simulated_annealing(depth_map_normalized, edge_map, objective_function)

# Normalize the optimized depth map to a range from 0 to 255 for saving
optimized_depth_map = cv2.normalize(optimized_depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Save the optimized depth map to a file
output_image_path = 'SA Image_Depth_Map.png'  # File name for the output image
cv2.imwrite(output_image_path, optimized_depth_map)

