import cv2
import numpy as np

# Read image
image = cv2.imread("image.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform watershed segmentation
_, markers = cv2.connectedComponents(np.uint8(gray))
markers = markers + 1
markers[gray == 255] = 0

# Apply watershed algorithm
markers = cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]

# Display result
cv2.imshow("Watershed Segmentation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

