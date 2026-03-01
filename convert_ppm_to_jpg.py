import os
import cv2
from pathlib import Path

def convert_ppm_to_jpg(dataset_path):
    print(f"Starting conversion in {dataset_path}...")
    converted_count = 0
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith('.ppm'):
                ppm_path = os.path.join(root, file)
                jpg_path = os.path.splitext(ppm_path)[0] + '.jpg'
                
                try:
                    # Read the ppm image
                    img = cv2.imread(ppm_path)
                    
                    if img is not None:
                        # Save it as jpg
                        cv2.imwrite(jpg_path, img)
                        
                        # Optionally remove the original ppm file to save space
                        # os.remove(ppm_path)
                        
                        converted_count += 1
                        if converted_count % 1000 == 0:
                            print(f"Converted {converted_count} images...")
                    else:
                        pr
                        int(f"Failed to read: {ppm_path}")
                except Exception as e:
                    print(f"Error converting {ppm_path}: {e}")
                    
    print(f"Finished! Successfully converted {converted_count} images.")

if __name__ == "__main__":
    # Point to the dataset directory
    dataset_dir = "d:/traffic sign classifier/dataset"
    convert_ppm_to_jpg(dataset_dir)
