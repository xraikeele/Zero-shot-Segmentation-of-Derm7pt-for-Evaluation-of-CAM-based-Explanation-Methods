import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def load_and_display_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    return image

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

def main():
    image_path = '/home/matthewcockayne/Documents/PhD/data/Derm7pt/release_v0/release_v0/images/A1l/Aal017.jpg'
    image = load_and_display_image(image_path)
    print(image.shape)

    sam_checkpoint = "/home/matthewcockayne/Documents/PhD/Models/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = initialize_sam(model_type, sam_checkpoint, device)
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(image)
    # Provide points as input prompt [X,Y]-coordinates
    input_point = np.array([[384, 256],[250,400]])
    input_label = np.array([1,0])


    # Predict the segmentation mask at that point
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
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

        # Check if the area of the mask is less than the total area of the image
        if score > max_score and mask_area < 0.7:  # adjust the threshold as needed
            max_score = score
            max_index = i

    # Save the image with the highest score
    if max_index != -1:
        mask_image = (masks[max_index] * 255).astype(np.uint8)
        file_name = f'mask_{max_index + 1}_max_score.png'
        cv2.imwrite(file_name, mask_image)

if __name__ == "__main__":
    main()