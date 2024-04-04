from PIL import Image
import numpy as np

np.set_printoptions(threshold=np.inf)


def process_image(image_path):
    # Load the image and convert it to greyscale
    img = Image.open(image_path).convert('L')
    
    # Convert the image into a numpy array for easier manipulation
    img_array = np.array(img)
    
    # Ensure the image is square
    height, width = img_array.shape
    if height != width:
        raise ValueError("The image must be square.")
    
    # Calculate the average pixel values across the diagonal mirror
    for x in range(width):
        for y in range(x, height):  # Start from x to only cover half above the diagonal
            # Mirror index calculation (swap x and y for the mirror pixel)
            mirror_x, mirror_y = y, x
            
            # Calculate the average of the current pixel and its diagonal mirror
            avg_pixel = (img_array[y, x] + img_array[mirror_x, mirror_y]) // 2
            
            # Set both pixels to the average value
            img_array[y, x] = avg_pixel
            img_array[mirror_x, mirror_y] = avg_pixel
    
    # Convert the numpy array back to an PIL image
    processed_img = Image.fromarray(img_array)
    
    # Return the processed image
    return processed_img



# # Example usage
# image_path = 'path_to_your_image.jpg'  # Update this to the path of your square image
# processed_image = process_image(image_path)
# processed_image.show()  # Display the processed image


ret = process_image("DatasetOrg/MNIST/mnist_png/train/1/2426.png")

ret.show()