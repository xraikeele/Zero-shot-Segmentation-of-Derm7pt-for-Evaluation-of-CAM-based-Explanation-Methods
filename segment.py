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
            
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_and_process_image(image_path):
    image = cv2.imread(image_path)
    # Create a subplot with two columns
    fig, axs = plt.subplots(4, 2, figsize=(10, 8))
    # Display original image
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')

    # Adjust image brightness
    enhanced_image = adjust_brightness_contrast(image, 1.2, 30)
    axs[0, 1].imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title('Enhanced Image')
    # Sharpen image
    sharpened_image = sharpen_image(enhanced_image)
    axs[1, 0].imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Sharpened Image')
    # Convert the image to grayscale
    gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
    axs[1, 1].imshow(gray, cmap='gray')
    axs[1, 1].set_title('Grayscale Image')
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Sobel operator
    img_sobel = cv2.Sobel(gray_blurred, cv2.CV_64F, 1, 0, ksize=5)
    axs[2, 0].imshow(gray_blurred, cmap='gray')
    axs[2, 0].set_title('Blurred Image')
    # Display Sobel gradient magnitude
    axs[2, 1].imshow(np.abs(img_sobel), cmap='gray')
    axs[2, 1].set_title('Sobel Gradient Magnitude')

    # Convert the Sobel result to a uint8 binary image
    sobel_binary = np.uint8(np.absolute(img_sobel))
    #_, threshold = cv2.threshold(sobel_binary, 128, 255, cv2.THRESH_BINARY)
    threshold = cv2.adaptiveThreshold(sobel_binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Display the binary image after thresholding
    axs[3, 1].imshow(threshold, cmap='gray')
    axs[3, 1].set_title('Binary Image after Sobel and Thresholding')
    # Adjust layout to prevent overlapping
    # Hide the empty subplot in the last row
    #axs[3, 1].axis('off')
    plt.tight_layout()
    plt.show()
    # Find contours in the binary image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    (centerXCoordinate, centerYCoordinate), radius = cv2.minEnclosingCircle(largest_contour)
    # Calculate the moments of the largest contour
    M = cv2.moments(largest_contour)

    # Calculate the centre of mass
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Convert centre coordinates to integers
    centerXCoordinate = int(centerXCoordinate)
    centerYCoordinate = int(centerYCoordinate)

    # Calculate the bounding box size based on the radius
    half_width = int(radius)
    half_height = int(radius)

    # Define the bounding box coordinates
    x1 = cX - half_width
    y1 = cY - half_height
    x2 = cX + half_width
    y2 = cY + half_height

    # Display the final result with the bounding box and centre point
    result_image = image.copy()
    cv2.circle(result_image, (centerXCoordinate, centerYCoordinate), int(radius), (255, 0, 0), 2)
    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Plot the centre point
    cv2.circle(result_image, (cX, cY), 5, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Final Result with Minimum Bounding Box and Center Point')
    plt.show()

    return sharpened_image, image, cX, cY, x1, y1, x2, y2

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
    sharpened_image, image, cX, cY, x1, y1, x2, y2 = load_and_process_image(image_path)
    print(image.shape)

    sam_checkpoint = "/home/matthewcockayne/Documents/PhD/Models/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = initialize_sam(model_type, sam_checkpoint, device)
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(sharpened_image)
    # Provide points as input prompt [X,Y]-coordinates
    input_point = np.array([[cX, cY],[(cX+100), cY],[(cX-100), cY],[(100),(100)]])
    #input_point = np.array([[cX, cY]])
    input_box = np.array([x1, y1, x2, y2])
    input_label = np.array([1,1,1,0])
    #input_label = np.array([1])


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