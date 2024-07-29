import os
import shutil
import random

dataset_dir = 'F:\\YOLOV8 project\\yolov8\\Automatic number plate\\automatic-number-plate-recognition-python\\yolov3-from-opencv-object-detection\\Dataset\\Dataset'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'yolo_labels')

train_images_dir = os.path.join(dataset_dir, 'train', 'images')
train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')
val_images_dir = os.path.join(dataset_dir, 'val', 'images')
val_labels_dir = os.path.join(dataset_dir, 'val', 'labels')

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

image_filenames = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

random.shuffle(image_filenames)

split_index = int(0.75 * len(image_filenames))
train_filenames = image_filenames[:split_index]
val_filenames = image_filenames[split_index:]

def copy_files(filenames, source_images_dir, source_labels_dir, target_images_dir, target_labels_dir):
    for filename in filenames:
    
        image_src_path = os.path.join(source_images_dir, filename)
        image_dest_path = os.path.join(target_images_dir, filename)
        shutil.copy(image_src_path, image_dest_path)
        label_filename = filename.rsplit('.', 1)[0] + '.txt'
        label_src_path = os.path.join(source_labels_dir, label_filename)
        label_dest_path = os.path.join(target_labels_dir, label_filename)
        if os.path.exists(label_src_path):  
            shutil.copy(label_src_path, label_dest_path)


copy_files(train_filenames, images_dir, labels_dir, train_images_dir, train_labels_dir)

copy_files(val_filenames, images_dir, labels_dir, val_images_dir, val_labels_dir)

print("Operatio complete")
