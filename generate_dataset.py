import os
import random
import json
import shutil
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance
from datasets import load_dataset

# Configuration
CLASSES = ["healthy", "maize_rust", "maize_blight", "cassava_mosaic", "bean_spot"]
SAMPLES_PER_CLASS = 300
IMAGE_SIZE = (224, 224)
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
FIELD_SAMPLES_TOTAL = 60

OUTPUT_DIR = "dataset_output"
MINI_PLANT_DIR = os.path.join(OUTPUT_DIR, "mini_plant_set")
TEST_FIELD_DIR = os.path.join(OUTPUT_DIR, "test_field")

def setup_directories():
    """Creates the necessary folder structure for the splits."""
    for split in SPLIT_RATIOS.keys():
        for cls in CLASSES:
            os.makedirs(os.path.join(MINI_PLANT_DIR, split, cls), exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(TEST_FIELD_DIR, cls), exist_ok=True)

def apply_field_augmentations(img):
    """Applies random blur, brightness jitter, and JPEG compression."""
    # Random Blur (sigma 0 to 1.5)
    sigma = random.uniform(0, 1.5)
    if sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    # Brightness Jitter
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # JPEG Compression (quality 50 to 85)
    q = random.randint(50, 85)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=q)
    return Image.open(buffer)

def process_and_save():
    setup_directories()
    
    class_weights = {}
    field_per_class = FIELD_SAMPLES_TOTAL // len(CLASSES)
    
    for cls in CLASSES:
        print(f"Processing {cls} ***")
        print(f"Downloading real images for {cls}...")
        
        total_needed = SAMPLES_PER_CLASS + field_per_class
        images = []
        
        # Load the real datasets from Hugging Face public mirrors
        if cls == "bean_spot":
            dataset = load_dataset("beans", split="train")
            # Label 0 is angular leaf spot in the beans dataset
            images = [item['image'] for item in dataset if item['labels'] == 0][:total_needed]
            
        elif cls == "cassava_mosaic":
            # Using an open benchmark mirror that requires no authentication
            dataset = load_dataset("dpdl-benchmark/cassava", split="train")
            # Label 3 is Cassava Mosaic Disease (CMD)
            images = [item['image'] for item in dataset if item['label'] == 3][:total_needed]
            
        else:
            # Fallback for the PlantVillage classes (healthy, maize_rust, maize_blight)
            # Using a reliable image-based mirror instead of a text-path mirror
            dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset", split="train")
            
            for item in dataset:
                label_val = item['label']
                
                # Extract the label name safely whether it's an integer ID or a raw string
                if isinstance(label_val, int):
                    label_name = dataset.features['label'].names[label_val].lower()
                else:
                    label_name = str(label_val).lower()
                
                # Dynamically match the specific maize diseases
                if "corn" in label_name or "maize" in label_name:
                    if cls == "healthy" and "healthy" in label_name:
                        images.append(item['image'])
                    elif cls == "maize_rust" and "rust" in label_name:
                        images.append(item['image'])
                    elif cls == "maize_blight" and "blight" in label_name:
                        images.append(item['image'])
                        
                # Stop iterating once we have enough images for the split and field sets
                if len(images) >= total_needed:
                    break
                    
        if len(images) < total_needed:
             raise ValueError(f"Not enough images pulled for {cls}. Got {len(images)}, need {total_needed}.")
        
        # Process clean set (80/10/10 split)
        clean_images = images[:SAMPLES_PER_CLASS]
        train_idx = int(SAMPLES_PER_CLASS * SPLIT_RATIOS["train"])
        val_idx = train_idx + int(SAMPLES_PER_CLASS * SPLIT_RATIOS["val"])
        
        splits = {
            "train": clean_images[:train_idx],
            "val": clean_images[train_idx:val_idx],
            "test": clean_images[val_idx:]
        }
        
        for split_name, split_imgs in splits.items():
            for i, img in enumerate(split_imgs):
                # Ensure RGB mode to prevent JPEG save errors
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_resized = img.resize(IMAGE_SIZE)
                save_path = os.path.join(MINI_PLANT_DIR, split_name, cls, f"{cls}_{i}.jpg")
                img_resized.save(save_path, format="JPEG")
                
        # Calculate naive class weights (balanced in this subset)
        class_weights[cls] = 1.0 
        
        # Process field robustness set
        field_images = images[SAMPLES_PER_CLASS:total_needed]
        for i, img in enumerate(field_images):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_resized = img.resize(IMAGE_SIZE)
            img_noisy = apply_field_augmentations(img_resized)
            save_path = os.path.join(TEST_FIELD_DIR, cls, f"{cls}_field_{i}.jpg")
            img_noisy.save(save_path, format="JPEG")

    # Generate README.md with class weights and provenance
    readme_content = f"""# Mini Plant Set
    - **Provenance**: Subsets pulled from public mirrors of PlantVillage, Cassava Leaf Disease, and Beans datasets.
    - **Total Images**: 1,500 clean, 60 noisy field-shots.
    - **Class Weights**: {json.dumps(class_weights, indent=2)}
    """
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
        f.write(readme_content)
    
    # Move README into the mini_plant_set directory so it gets zipped together
    shutil.copy(os.path.join(OUTPUT_DIR, "README.md"), os.path.join(MINI_PLANT_DIR, "README.md"))

def package_zips():
    """Zips the directories using pure Python for Colab/Windows compatibility."""
    print("Zipping mini_plant_set.zip...")
    # This will create mini_plant_set.zip in your current working directory
    shutil.make_archive("mini_plant_set", "zip", OUTPUT_DIR, "mini_plant_set")
    
    print("Zipping test_field.zip...")
    # This will create test_field.zip in your current working directory
    shutil.make_archive("test_field", "zip", OUTPUT_DIR, "test_field")

if __name__ == "__main__":
    process_and_save()
    package_zips()
    print("Dataset generation complete. Zips ready for download!")