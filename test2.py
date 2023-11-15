import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('/home/matthewcockayne/Documents/PhD/data/Derm7pt/release_v0/release_v0/images/A1l/Aal017.jpg')

# Display the original image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.show()

# Apply CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(gray)

# Display the enhanced image
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')
plt.show()

# Apply Canny edge detection
edges = cv2.Canny(enhanced_image, 30, 100)

# Display the edges
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.show()

# Apply morphological operations
kernel = np.ones((5, 5), np.uint8)
morph_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Display the morphological image
plt.imshow(morph_image, cmap='gray')
plt.title('Morphological Operations')
plt.show()

# Find contours in the binary image
contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over all contours
for contour in contours:
    # Find the contour area
    area = cv2.contourArea(contour)

    # Filter contours based on area
    if area > 500:  # Adjust the threshold as needed
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)

        # Calculate the moments of the current contour
        M = cv2.moments(approx_contour)

        # Calculate the centre of mass
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Draw the centre and contour on the image
        cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)
        cv2.drawContours(image, [approx_contour], 0, (0, 255, 0), 2)

# Display the final result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Final Result with Centre Point Detection and Bounding Box')
plt.show()