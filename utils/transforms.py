import albumentations as A
import numpy as np
import cv2

from albumentations.pytorch import ToTensorV2
from torchvision import transforms as transforms

def resize(im, img_size=640, square=False):
    # Aspect ratio resize
    if square:
        im = cv2.resize(im, (img_size, img_size))
    else:
        h0, w0 = im.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
    return im

# Define the training tranforms
def get_train_aug():
    return A.Compose([
        # FLIP sangat aman dan menaikkan mAP untuk semua ukuran
        A.HorizontalFlip(p=0.5),
        # --- SCALE / CROP untuk small object ---
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.10,      # overscale membantu small
            rotate_limit=5,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.35
        ),
        # Small object enhancer (tanpa merusak bbox)
        A.RandomResizedCrop(
            height=640, width=640,
            scale=(0.85, 1.0),     # crop ringan → bantu small
            ratio=(0.9, 1.1),
            p=0.25
        ),
        # --- BLUR / SHARP untuk medium-large ---
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.MedianBlur(blur_limit=3, p=0.2),
        ], p=0.45),
        # --- KONDISI API & ASAP ---
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.25,
                p=0.6
            ),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.015,
                p=0.4
            ),
        ], p=0.5),
        # Fog/Haze meningkatkan kemampuan mendeteksi smoke
        A.RandomFog(
            alpha_coef=0.04,
            fog_coef_lower=0.1,
            fog_coef_upper=0.3,
            p=0.20
        ),
        # --- GAMMA untuk objek besar (api lebih terlihat) ---
        A.RandomGamma(
            gamma_limit=(85, 120),
            p=0.25
        ),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.20
    ))


def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))

def transform_mosaic(mosaic, boxes, img_size=640):
    """
    Resizes the `mosaic` image to `img_size` which is the desired image size
    for the neural network input. Also transforms the `boxes` according to the
    `img_size`.

    :param mosaic: The mosaic image, Numpy array.
    :param boxes: Boxes Numpy.
    :param img_resize: Desired resize.
    """
    aug = A.Compose(
        [A.Resize(img_size, img_size, always_apply=True, p=1.0)
    ])
    sample = aug(image=mosaic)
    resized_mosaic = sample['image']
    transformed_boxes = (np.array(boxes) / mosaic.shape[0]) * resized_mosaic.shape[1]
    for box in transformed_boxes:
        # Bind all boxes to correct values. This should work correctly most of
        # of the time. There will be edge cases thought where this code will
        # mess things up. The best thing is to prepare the dataset as well as 
        # as possible.
        if box[2] - box[0] <= 1.0:
            box[2] = box[2] + (1.0 - (box[2] - box[0]))
            if box[2] >= float(resized_mosaic.shape[1]):
                box[2] = float(resized_mosaic.shape[1])
        if box[3] - box[1] <= 1.0:
            box[3] = box[3] + (1.0 - (box[3] - box[1]))
            if box[3] >= float(resized_mosaic.shape[0]):
                box[3] = float(resized_mosaic.shape[0])
    return resized_mosaic, transformed_boxes

# Define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
    ))

def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)
