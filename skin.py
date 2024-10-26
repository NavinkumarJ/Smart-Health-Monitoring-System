import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_skin(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the HSV range for skin color
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask where skin colors are white
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Apply the mask to the original image
    skin = cv2.bitwise_and(image, image, mask=mask)

    return skin, mask

def detect_pimples(mask):
    # Find contours of the masked areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours that could be noise
    min_contour_area = 100
    pimples = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    return pimples

def draw_contours(image, contours):
    # Draw contours on the original image
    for contour in contours:
        cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

    return image

# Load the image
image = cv2.imread("pimple.jpeg")  # Replace 'face_image.jpg' with your image file

# Detect skin in the image
skin, mask = detect_skin(image)

# Detect pimples or blemishes
pimples = detect_pimples(mask)

# Draw contours around detected pimples
result_image = draw_contours(image.copy(), pimples)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
plt.title('Detected Skin')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('Detected Pimples')
plt.show()
