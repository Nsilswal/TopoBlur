import numpy as np
import cv2

def load_image(image_path):
    # Load an image as greyscale
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def make_symmetric(image):
    # Ensure the image is square
    assert image.shape[0] == image.shape[1], "Image must be square"
    
    n = image.shape[0]  # Get the dimensions of the image
    symmetric_image = image.copy()  # Create a copy of the image to modify
    
    for i in range(n):
        for j in range(i+1, n):
            # Perform the addition in a larger data type to avoid overflow
            avg_pixel_value = (image[i, j].astype(int) + image[j, i].astype(int)) // 2
            symmetric_image[i, j] = avg_pixel_value
            symmetric_image[j, i] = avg_pixel_value
    
    return symmetric_image


def save_image(image, output_path):
    # Save the modified image
    cv2.imwrite(output_path, image)

# Example usage
image_path = "DatasetOrg/MNIST/mnist_png/train/8/17.png"
output_path = "symmetric_image_8_17.jpg"

# For my initial pipelining processes, I'm going to use train/8/17.png as the image to be processed

# Load the image
image = load_image(image_path)

# Make the image symmetric along the diagonal
symmetric_image = make_symmetric(image)

# print(symmetric_image)

save_image(symmetric_image, output_path)

cv2.imshow('Symmetric Image', symmetric_image)

# Wait for a key press and then close the displayed windows
cv2.waitKey(0)
cv2.destroyAllWindows()
