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

def main():
    image_path = '/home/matthewcockayne/Documents/PhD/segment-anything/notebooks/images/Nal047.jpg'
    image = load_and_display_image(image_path)

    sam_checkpoint = "/home/matthewcockayne/Documents/PhD/Models/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = initialize_sam(model_type, sam_checkpoint, device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = generate_masks(image, mask_generator)

    print(len(masks))
    print(masks[0].keys())

    _,axes = plt.subplots(1,2, figsize=(16,16))
    axes[0].imshow(image)
    show_output(masks, axes[1])
    #plt.figure(figsize=(20,20))
    #plt.imshow(image)
    #show_anns(masks)
    #plt.axis('off')
    #plt.show()

if __name__ == "__main__":
    main()