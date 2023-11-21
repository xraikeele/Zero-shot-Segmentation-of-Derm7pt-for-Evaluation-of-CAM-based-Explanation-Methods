import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def adjust_brightness_contrast(image, alpha, beta):
    return cv2.addWeighted(image, alpha, image, 0, beta)
            
def load_and_process_image(image_path):
    # Read the original image
    image = cv2.imread(image_path)
    # Crop image
    #crop_img = image[80:280, 150:330]
    #cv2.imshow(crop_img)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray)

    # Apply Canny edge detection
    edges = cv2.Canny(enhanced_image, 30, 100)

    # Apply adaptive thresholding
    adaptive_threshold = cv2.adaptiveThreshold(enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)

    # Combine Canny edges and adaptive threshold
    combined_image = cv2.bitwise_or(edges, adaptive_threshold)

    # Apply morphological operations to close gaps
    kernel = np.ones((5, 5), np.uint8)
    morph_image = cv2.morphologyEx(combined_image, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary image
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Dynamic area threshold based on image size
    area_threshold = 0.2 * image.size

    # Initialize bounding box coordinates
    x1, y1, x2, y2 = 0, 0, 0, 0
    
    # Iterate over all contours
    for contour in contours:
        # Find the convex hull of the contour
        hull = cv2.convexHull(contour)

        # Get the bounding box of the convex hull
        x, y, w, h = cv2.boundingRect(hull)

        # Filter contours based on area
        if w * h > area_threshold:
            # Set bounding box coordinates
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Draw the bounding box on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Optionally, you can calculate the center of mass
            M = cv2.moments(hull)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Draw the center of mass
                cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)
            else:
                cX, cY = 0, 0

    # Display the results
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Processed Image with Convex Hull and Bounding Box')
    plt.show()

    return image, cX, cY, x1, y1, x2, y2

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def initialize_sam(model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)

    return sam

def generate_masks(image, mask_generator):
    masks = mask_generator.generate(image)

    return masks

def show_output(result_dict, axes=None):
    if axes:
        ax = axes
    else:
        _, ax = plt.subplots(figsize=(8, 8))
        ax.set_autoscale_on(False)

    sorted_result = sorted(result_dict, key=(lambda x: x['area']), reverse=True)

    # Plot for each segment area
    for val in sorted_result:
        mask = val['segmentation']
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]

        for i in range(3):
            img[:, :, i] = color_mask[i]

        ax.imshow(np.dstack((img, mask * 0.5)))

    plt.show()

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

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def main():
    image_path = '/home/matthewcockayne/Documents/PhD/data/Derm7pt/release_v0/release_v0/images/A1l/Aal017.jpg'
    image, cX, cY,  x1, y1, x2, y2 = load_and_process_image(image_path)
    print(image.shape)

    sam_checkpoint = "/home/matthewcockayne/Documents/PhD/Models/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = initialize_sam(model_type, sam_checkpoint, device)
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(image)
    # Provide points as input prompt [X,Y]-coordinates
    #input_point = np.array([[cX, cY],[(cX+100), cY],[cX, (cY+100)],[(100),(100)]])
    input_point = np.array([[cX, cY],[(100),(100)]])
    input_box = np.array([x1, y1, x2, y2])
    #input_label = np.array([1,1,1,0])
    input_label = np.array([1,0])


    # Predict the segmentation mask at that point
    masks, scores, logits = mask_predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=True,
    )
    max_score = -1
    max_index = -1

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_area = mask.sum() / mask.size
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        show_box(input_box, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

        # Check if the area of the mask is less than the total area of the image
        if score > max_score and mask_area < 0.6:  # adjust the threshold as needed
            max_score = score
            max_index = i

    # Save the image with the highest score
    if max_index != -1:
        mask_image = (masks[max_index] * 255).astype(np.uint8)
        file_name = f'mask_{max_index + 1}_max_score.png'
        cv2.imwrite(file_name, mask_image)

if __name__ == "__main__":
    main()