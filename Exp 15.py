import cv2
import numpy as np

# Read image
image = cv2.imread("image.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform segmentation (e.g., thresholding)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Display original and segmented images
cv2.imshow("Original Image", image)
cv2.imshow("Segmented Image", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

