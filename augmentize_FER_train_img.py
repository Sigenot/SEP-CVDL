import os
from torchvision import transforms
from PIL import Image

# define transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # horizontal flip with a probability of 50%
    transforms.RandomRotation(30), # random rotation of 30 degrees
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1) # random changes for brightness, contrast, saturation and hue
])

transform_disg = transforms.Compose([ # no random rotation this time, to ensure a realistic orientation of the faces
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
])

# main dataset folder and train/test sets
dataset_dir = 'dataset2_raf-db'
base_dirs = ['train']

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

# iterate main folders
for base_dir in base_dirs:
    for emotion in emotions:
        # path to current emotion folder
        folder_path = os.path.join(dataset_dir, base_dir, emotion)

        # check if path exists
        if not os.path.exists(folder_path):
            continue

        # list of all images in current folder
        images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # iterate each image
        for image_name in images:
            image_path = os.path.join(folder_path, image_name)
            image = Image.open(image_path)

            # augmentize
            augmented_image = transform(image)

            # new name for augmented image to ensure human recognizability
            new_image_name = f'aug_{image_name}'
            new_image_path = os.path.join(folder_path, new_image_name)

            # save augmented image
            augmented_image.save(new_image_path)


emotions = ['disgust']

# iterate main folders
# counter
i = 0
while i <= 4:
    for base_dir in base_dirs:
        for emotion in emotions:
            # path to current emotion folder
            folder_path = os.path.join(dataset_dir, base_dir, emotion)

            # check if path exists
            if not os.path.exists(folder_path):
                continue

            # list of all images in current folder
            images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

            # iterate each image
            for image_name in images:
                image_path = os.path.join(folder_path, image_name)
                image = Image.open(image_path)

                # augmentize
                augmented_image = transform(image)

                # new name for augmented image to ensure human recognizability
                new_image_name = f'more_aug_{image_name}'
                new_image_path = os.path.join(folder_path, new_image_name)

                # save augmented image
                augmented_image.save(new_image_path)
    i += 1

"""
Numbers after augmentation dataset1:
angry: 11985
disgust: 9156
fear: 12291
happy: 21645 --> probably still leaning heavily to happy
sad: 14490
surprise: 9513
"""

"""
Numbers after augmentation dataset2:
angry: 1410
disgust: 8604
fear: 562
happy: 9544
sad: 3964
surprise: 2580
"""