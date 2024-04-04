import cv2
import os
from PIL import Image

def modify_image(image_path, alpha, beta):
    image = cv2.imread(image_path)
    
    # Apply Gaussian Blur
    if beta > 0:
        blurred_image = cv2.GaussianBlur(image, (0, 0), beta)
    else:
        blurred_image = image

    # Convert to PIL Image for easier manipulation
    image_pil = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))

    if alpha != 0:
        # Add or remove padding to make the image square
        old_size = image_pil.size  # old_size[0] is in (width, height) format

        if alpha > 0:
            # Add padding
            new_size = tuple(2*max(old_size) for _ in old_size)
            new_im = Image.new("RGB", new_size)
            new_im.paste(image_pil, (int((new_size[0]-old_size[0])/2), int((new_size[1]-old_size[1])/2)))
        else:
            # Remove padding
            crop_size = tuple(max(old_size) - abs(alpha) for _ in old_size)
            new_im = image_pil.crop((abs(alpha)/2, abs(alpha)/2, crop_size[0], crop_size[1]))
    else:
        new_im = image_pil

    return new_im

def process_folder(folder_path, alpha, beta):
    # Ensure the output directory exists
    output_dir = os.path.join(folder_path, "modified")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            modified_image = modify_image(file_path, alpha, beta)
            # Save the modified image
            save_path = os.path.join(output_dir, filename)
            modified_image.save(save_path)
            print(f"Processed and saved: {save_path}")

# Example usage
folder_path = "DatasetOrg/Test/"
alpha = 50  # Change as needed
beta = 1.5  # Change as needed
process_folder(folder_path, alpha, beta)
