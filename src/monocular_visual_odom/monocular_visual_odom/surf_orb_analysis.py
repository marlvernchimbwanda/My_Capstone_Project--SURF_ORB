import os
import cv2
import time
import numpy as np
import psutil
import matplotlib.pyplot as plt

# Function to measure memory usage
def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 ** 2)  # Convert to MB

# Load images
root = os.getcwd()
img_path_1 = os.path.join(root, 'datasets/KITTI_sequence_2/image_l/000049.png')
img_path_2 = os.path.join(root, 'datasets/KITTI_sequence_2/image_l/000050.png')

img1 = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img_path_2, cv2.IMREAD_GRAYSCALE)

# Ensure images are loaded
if img1 is None or img2 is None:
    raise FileNotFoundError("Images could not be loaded. Check the paths.")

# 1st Approach: SURF Detector + ORB Descriptor
try:
    surf = cv2.xfeatures2d.SURF_create(3000)
except AttributeError:
    raise Exception("SURF not found. Install OpenCV with contrib modules: `pip install opencv-contrib-python`")

orb = cv2.ORB_create(3000)

# Measure memory before processing
mem_before_surf_orb = get_memory_usage()

# Detect keypoints with SURF
start_time = time.time()
kp1_surf = surf.detect(img1, None)
kp2_surf = surf.detect(img2, None)


# Compute ORB descriptors
kp1_surf, des1_surf = orb.compute(img1, kp1_surf)
kp2_surf, des2_surf = orb.compute(img2, kp2_surf)

# Measure memory after processing
surf_orb_time = time.time() - start_time
mem_after_surf_orb = get_memory_usage()

mem_before_orb = get_memory_usage()
start_time = time.time()
kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
kp2_orb, des2_orb = orb.detectAndCompute(img2, None)
orb_time = time.time() - start_time
mem_after_orb = get_memory_usage()

# Matching using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match features
matches_surf_orb = bf.match(des1_surf, des2_surf)
matches_orb = bf.match(des1_orb, des2_orb)

# Sort matches by distance
matches_surf_orb = sorted(matches_surf_orb, key=lambda x: x.distance)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

# Calculate memory usage differences
mem_usage_surf = mem_after_surf_orb - mem_before_surf_orb
mem_usage_orb = mem_after_orb - mem_before_orb

# Display results
print(f"SURF+ORB Time: {surf_orb_time:.4f} seconds")
print(f"ORB+ORB Time: {orb_time:.4f} seconds")
print(f"SURF+ORB Keypoints: {len(kp1_surf)}")
print(f"ORB+ORB Keypoints: {len(kp1_orb)}")
print(f"SURF+ORB Matches: {len(matches_surf_orb)}")
print(f"ORB+ORB Matches: {len(matches_orb)}")
print(f"SURF+ORB Memory Usage: {mem_usage_surf:.4f} MB")
print(f"ORB+ORB Memory Usage: {mem_usage_orb:.4f} MB")

# Visualize matches
img_matches_surf_orb = cv2.drawMatches(img1, kp1_surf, img2, kp2_surf, matches_surf_orb[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches_orb = cv2.drawMatches(img1, kp1_orb, img2, kp2_orb, matches_orb[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Plot efficiency comparisons
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Keypoints comparison
axs[0, 0].bar(["SURF+ORB", "ORB+ORB"], [len(kp1_surf), len(kp1_orb)], color=['blue', 'green'])
axs[0, 0].set_title("Number of Keypoints Detected")

# Time comparison
axs[0, 1].bar(["SURF+ORB", "ORB+ORB"], [surf_orb_time, orb_time], color=['red', 'purple'])
axs[0, 1].set_title("Time Taken (seconds)")

# Matches comparison
axs[1, 0].bar(["SURF+ORB", "ORB+ORB"], [len(matches_surf_orb), len(matches_orb)], color=['orange', 'cyan'])
axs[1, 0].set_title("Number of Matches Found")

# Memory comparison
axs[1, 1].bar(["SURF+ORB", "ORB+ORB"], [mem_usage_surf, mem_usage_orb], color=['brown', 'pink'])
axs[1, 1].set_title("Memory Usage (MB)")

# Display images with matches
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(img_matches_surf_orb)
plt.title("SURF Detector + ORB Descriptor")

plt.subplot(1, 2, 2)
plt.imshow(img_matches_orb)
plt.title("ORB Detector + ORB Descriptor")

plt.show()
