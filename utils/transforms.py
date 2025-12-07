import albumentations as A
import numpy as np
import cv2
import random

from albumentations.core.transforms_interface import DualTransform
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

class CopyPasteCustom(DualTransform):
    def __init__(self, blend=True, sigma=1, pct_objects_paste=0.5, always_apply=False, p=0.5):
        super(CopyPasteCustom, self).__init__(always_apply, p)
        self.blend = blend
        self.sigma = sigma
        self.pct_objects_paste = pct_objects_paste

    def apply(self, img, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_params_dependent_on_targets(self, params):
        return {}

    @property
    def targets_as_params(self):
        return ["image", "bboxes", "labels"]

    def update_params(self, params, **kwargs):
        return params

    def apply_with_params(self, params, **kwargs):
        return kwargs["image"], kwargs["bboxes"], kwargs["labels"]

    def __call__(self, force_apply=False, **kwargs):
        if not force_apply and random.random() > self.p:
            return kwargs

        image = kwargs["image"].copy()
        bboxes = kwargs["bboxes"].copy()
        labels = kwargs["labels"].copy()

        if len(bboxes) == 0:
            return kwargs  # nothing to paste

        # pilih subset objek
        n = max(1, int(len(bboxes) * self.pct_objects_paste))
        idxs = random.sample(range(len(bboxes)), n)

        for idx in idxs:
            bbox = bboxes[idx][:4]
            x1, y1, x2, y2 = map(int, bbox)
            obj = image[y1:y2, x1:x2].copy()

            # posisi baru
            h, w = image.shape[:2]
            new_x = random.randint(0, w - (x2 - x1))
            new_y = random.randint(0, h - (y2 - y1))
            new_x2 = new_x + (x2 - x1)
            new_y2 = new_y + (y2 - y1)

            # blending
            if self.blend:
                mask = np.full(obj.shape[:2], 255, np.uint8)
                mask = cv2.GaussianBlur(mask, (0, 0), self.sigma)
                mask = mask[..., None] / 255.0
                image[new_y:new_y2, new_x:new_x2] = \
                    obj * mask + image[new_y:new_y2, new_x:new_x2] * (1 - mask)
            else:
                image[new_y:new_y2, new_x:new_x2] = obj

            bboxes.append([new_x, new_y, new_x2, new_y2])
            labels.append(labels[idx])

        kwargs["image"] = image
        kwargs["bboxes"] = [bb[:4] for bb in bboxes]
        kwargs["labels"] = labels
        return kwargs

# Define the training tranforms
def get_train_aug():
    return A.Compose([
        A.RandomSizedBBoxSafeCrop(
            height=512, width=512,
            p=0.30
        ),
        A.CoarseDropout(
            max_holes=6,
            max_height=48,
            max_width=48,
            min_holes=1,
            min_height=24,
            min_width=24,
            fill_value=0,
            p=0.25
        ),
        CopyPasteCustom(
            blend=True,
            sigma=1,
            pct_objects_paste=0.5,
            p=0.50
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.10,     
            rotate_limit=7,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.40
        ),
        A.RandomResizedCrop(
            height=640, width=640,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=0.15
        ),
        A.MotionBlur(p=0.25),
        A.GaussianBlur(p=0.25),
        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.20,
            p=0.50
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
            p=0.40
        ),
        A.RandomFog(p=0.10),
        A.RandomGamma(p=0.20),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.10,
        min_area=16
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
