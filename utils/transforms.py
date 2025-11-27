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
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": 0.05, "y": 0.05},
            scale=(0.9, 1.1),
            rotate=(-5, 5),
            fit_output=False,
            mode=0,
            p=0.3
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.4),
            A.GaussianBlur(blur_limit=(3,5), p=0.3),
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02, p=0.2),
        A.RandomFog(alpha_coef=0.04, p=0.15),
        A.RandomSunFlare(
            flare_roi=(0,0,1,0.5),
            src_radius=120,
            p=0.15
        ),
        A.RandomShadow(p=0.15),
        A.CLAHE(p=0.1),
        A.RandomGamma(gamma_limit=(90,110), p=0.15),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
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
