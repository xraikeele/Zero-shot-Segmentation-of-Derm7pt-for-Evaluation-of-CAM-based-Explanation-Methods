import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def load_and_process_image(image_path):
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a threshold to create a binary image
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Iterate through the contours
    for contour in contours:
        # Calculate the moments of the contour
        M = cv2.moments(contour)

        # Calculate the centre of mass
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

    return image, cX, cY

def initialize_sam(model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)

    return sam

def generate_masks(image, mask_generator):
    masks = mask_generator.generate(image)

    return masks

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def process_images_in_directory(input_directory, output_directory, batch_size=10):
    os.makedirs(output_directory, exist_ok=True)

    image_paths = []
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                print(f"Processing image: {image_path}")
                image_paths.append(image_path)

    num_images = len(image_paths)

    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch_paths = image_paths[start_idx:end_idx]

        process_batch(batch_paths, output_directory)

def process_batch(image_paths, output_directory):
    for image_path in image_paths:
        # Load the image and cX, cY 
        image, cX, cY = load_and_process_image(image_path)
        
        output_path = os.path.join(output_directory, f"{os.path.splitext(os.path.basename(image_path))[0]}_mask.png")

        process_single_image(image, output_path, cX, cY)

def process_single_image(image, output_path, cX, cY):
    sam_checkpoint = "/home/matthewcockayne/Documents/PhD/Models/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = initialize_sam(model_type, sam_checkpoint, device)
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(image)
    input_point = np.array([[cX, cY]])
    input_label = np.array([1])

    masks, scores, logits = mask_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    max_score = -1
    max_index = -1

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_area = mask.sum() / mask.size
        plt.figure(figsize=(10, 10))
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.close()

        if score > max_score and mask_area < 0.7:
            max_score = score
            max_index = i

    if max_index != -1:
        mask_image = (masks[max_index] * 255).astype(np.uint8)
        cv2.imwrite(output_path, mask_image)

if __name__ == "__main__":
    input_directory = '/home/matthewcockayne/Documents/PhD/data/Derm7pt/release_v0/release_v0/images/A2l'
    output_directory = '/home/matthewcockayne/Documents/PhD/data/Derm7pt/release_v0/release_v0/masks/A2l'

    process_images_in_directory(input_directory, output_directory)