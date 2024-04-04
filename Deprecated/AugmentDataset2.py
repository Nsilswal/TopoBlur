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

def process_folder(folder_path, alpha, beta, output_root):
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, name)
                modified_image = modify_image(file_path, alpha, beta)
                
                # Construct a corresponding output path that mirrors the input hierarchy
                relative_path = os.path.relpath(root, folder_path)
                output_dir = os.path.join(output_root, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                save_path = os.path.join(output_dir, name)
                modified_image.save(save_path)
                print(f"Processed and saved: {save_path}")

# Define your dataset paths
original_dataset_path = "DatasetOrg/MNIST/mnist_png"
output_dataset_root = "MNIST_modified_no_padding"

alpha = 0  # Change as needed
beta = 4  # Change as needed

# Process both the train and test folders within the dataset
for subfolder in ['train', 'test']:
    folder_path = os.path.join(original_dataset_path, subfolder)
    process_folder(folder_path, alpha, beta, output_dataset_root)
