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
    sharpened = cv2.filter2D(image, -1, kernel)
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.title('Sharpened Image')
    plt.show()
    return sharpened

def adjust_brightness_contrast(image, alpha, beta):
    adjusted = cv2.addWeighted(image, alpha, image, 0, beta)
    plt.imshow(cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB))
    plt.title('Adjusted Brightness/Contrast Image')
    plt.show()
    return adjusted

def load_and_process_image(image_path):
    image = cv2.imread(image_path)

    # Display original image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.show()

    # Adjust image brightness
    enhanced_image = adjust_brightness_contrast(image, 1.2, 30)

    # Sharpen image
    sharpened_image = sharpen_image(enhanced_image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.show()

    # Apply Sobel operator
    img_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    
    # Display Sobel gradient magnitude
    plt.imshow(np.abs(img_sobel), cmap='gray')
    plt.title('Sobel Gradient Magnitude')
    plt.show()

    # Convert the Sobel result to a uint8 binary image
    sobel_binary = np.uint8(np.absolute(img_sobel))
    _, threshold = cv2.threshold(sobel_binary, 128, 255, cv2.THRESH_BINARY)

    # Display the binary image after thresholding
    plt.imshow(threshold, cmap='gray')
    plt.title('Binary Image after Sobel and Thresholding')
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

    # Define the bounding box coordinates
    x1 = cX - 100
    y1 = cY - 100
    x2 = cX + 100
    y2 = cY + 100

    # Display the final result with the bounding box
    result_image = image.copy()
    cv2.drawContours(result_image, [largest_contour], 0, (0, 255, 0), 2)
    cv2.circle(result_image, (centerXCoordinate, centerYCoordinate), radius, (255,0,0), 2)
    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Final Result with Bounding Box')
    plt.show()

    return image, cX, cY, x1, y1, x2, y2

# Example usage
image_path = '/home/matthewcockayne/Documents/PhD/data/Derm7pt/release_v0/release_v0/images/A1l/Aal017.jpg'
image, cX, cY, x1, y1, x2, y2 = load_and_process_image(image_path)