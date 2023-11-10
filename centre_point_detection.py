import cv2
import numpy as np

# Read the image
image = cv2.imread('/home/matthewcockayne/Documents/PhD/data/Derm7pt/release_v0/release_v0/images/NAL/Nal036.jpg')

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

    # Draw the centre and contour on the image
    cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)
    cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)

# Display the result
cv2.imshow('Centre Point Detection', image)
# Save the processed image
cv2.imwrite('/home/matthewcockayne/Documents/PhD/Zero-shot-Segmentation-of-Derm7pt-for-Evaluation-of-CAM-based-Explanation-Methods/figures/centre_point_detection3.jpg', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()