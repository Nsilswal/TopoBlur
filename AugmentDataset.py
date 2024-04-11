import cv2
import os
from PIL import Image
import random

def modify_image(image_path, alpha, beta):
    image = cv2.imread(image_path)
    
    # Apply Gaussian Blur with randomness when beta > 0
    if beta > 0:
        # Generate random sigmaX and sigmaY based on beta
        sigmaX = random.uniform(beta / 2, beta * 1.5)
        sigmaY = random.uniform(beta / 2, beta * 1.5)
        # Kernel size is automatically determined from sigmaX and sigmaY but must be odd
        ksizeX = int(6 * sigmaX + 1) | 1
        ksizeY = int(6 * sigmaY + 1) | 1
        blurred_image = cv2.GaussianBlur(image, (ksizeX, ksizeY), sigmaX, sigmaY)
    else:
        blurred_image = image

    # Convert to PIL Image for easier manipulation
    image_pil = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
    
    if alpha != 0:
        width, height = image_pil.size
        operation = random.choice(['stretch_v', 'stretch_h', 'compress_v', 'compress_h'])
        
        if operation == 'stretch_v':
            new_height = int(height * (1 + abs(alpha) / 100))
            new_size = (width, new_height)
        elif operation == 'stretch_h':
            new_width = int(width * (1 + abs(alpha) / 100))
            new_size = (new_width, height)
        elif operation == 'compress_v':
            new_height = int(height / (1 + abs(alpha) / 100))
            new_size = (width, max(new_height, 1))  # Ensure new height is at least 1
        else:  # compress_h
            new_width = int(width / (1 + abs(alpha) / 100))
            new_size = (max(new_width, 1), height)  # Ensure new width is at least 1
        
        new_im = image_pil.resize(new_size, Image.ANTIALIAS)
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
output_dataset_root = "MNIST_blur_beta2"

alpha = 0  # Change as needed degree of stretch compress
beta = 3  # Change as needed degree of blur 

# Process both the train and test folders within the dataset
for subfolder in ['train', 'test']:
    folder_path = os.path.join(original_dataset_path, subfolder)
    process_folder(folder_path, alpha, beta, output_dataset_root)
