import os
import cv2
import random
import albumentations as A

def augment_dataset(base_folder, n_per_person):
    # pipeline ของการทำ augmentation
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.RandomGamma(p=0.3),
        A.CLAHE(p=0.3),
    ])

    persons = [p for p in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, p))]

    for person in persons:
        person_path = os.path.join(base_folder, person)
        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"\n👤 Processing: {person} (found {len(images)} images)")

        count = 0
        while count < n_per_person:
            img_name = random.choice(images)
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            # ทำ Augmentation
            augmented = transform(image=image)
            aug_img = augmented["image"]

            new_name = f"{person}_aug_{count}.jpg"
            cv2.imwrite(os.path.join(person_path, new_name), aug_img)
            count += 1

        print(f"✅ Created {n_per_person} augmented images for {person}")

# ------------------ ตัวอย่างการใช้งาน ------------------
base_folder = "datasets"
n_per_person = 20
augment_dataset(base_folder, n_per_person)