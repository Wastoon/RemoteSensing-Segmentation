import torch
import torch.nn as nn
from tqdm import tqdm
import glob
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from utils import load_checkpoint
import cv2
from model import get_model

###Hyperparameter
TEST_DATA_DIR = '/tmp/pycharm_project_677/data/origins/suichang_round1_test_partA_210120/suichang_round1_test_partA_210120'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHECKPOINT = 'best_checkpoint.pth.tar'


def test(model, test_data_dir, transform=None, device='cuda'):
    model.eval()

    for idx, name in enumerate(tqdm(glob.glob(os.path.join(test_data_dir, '*.tif'))[:])):
        im = Image.open(name)
        image_RGB = np.array(im)[:, :, :-1]
        if transform is not None:
            image_RGB = transform(image=image_RGB)
            image_RGB = image_RGB['image']

        with torch.no_grad():
            image_RGB = image_RGB.unsqueeze(0).to(device=device)
            score = model(image_RGB).clone().cpu().softmax(1).numpy()

            score_sigmoid = score[0].argmax(0) + 1
            assert score_sigmoid.max() <= 10, "category pred is wrong, bigger than 10"
            assert score_sigmoid.min() >= 1, "category pred is wrong, smaller than 1"

            cv2.imwrite('results/' + os.path.basename(name).replace('.tif', '.png'), score_sigmoid)


def complicated_test(model, test_data_dir, transform_dict=None, device='cuda'):
    model.eval()

    for idx, name in enumerate(tqdm(glob.glob(os.path.join(test_data_dir, '*.tif'))[:])):
        im = Image.open(name)
        image_org = np.array(im)[:, :, :-1]
        if transform_dict is not None:
            score_list = []
            res = 0

            for method, transform in transform_dict.items():
                image_RGB = transform(image=image_org)
                image_RGB = image_RGB['image']
                with torch.no_grad():
                    image_RGB = image_RGB.unsqueeze(0).to(device=device)
                    score = model(image_RGB).clone().cpu().numpy()[0]

                    if method == 'HorizontalFlip':
                        score = np.flip(score, 2)

                    if method == 'VerticalFlip':
                        score = np.flip(score, 1)

                    if method == 'Transpose':
                        score = score.transpose((0, 2, 1))

                    score_list.append(score)

            for score in score_list:
                res = res + score
            res = res / len(score_list)
            score_sigmoid = res.argmax(0) + 1
            assert score_sigmoid.max() <= 10, "category pred is wrong, bigger than 10"
            assert score_sigmoid.min() >= 1, "category pred is wrong, smaller than 1"

            cv2.imwrite('results/' + os.path.basename(name).replace('.tif', '.png'), score_sigmoid)


if __name__ == "__main__":
    model = get_model().to(device=DEVICE)

    load_checkpoint(torch.load(CHECKPOINT), model)

    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.625, 0.448, 0.688],
                std=[0.131, 0.177, 0.101],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    test_complicated_transfom1 = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[48.5125, 60.0272, 56.5723],
                std=[32.3620, 31.1285, 30.2979],
                max_pixel_value=255.0
            ),
            A.HorizontalFlip(p=1),
            ToTensorV2(),
        ]
    )

    test_complicated_transfom2 = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[48.5125, 60.0272, 56.5723],
                std=[32.3620, 31.1285, 30.2979],
                max_pixel_value=255.0
            ),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ]
    )
    test_complicated_transfom3 = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[48.5125, 60.0272, 56.5723],
                std=[32.3620, 31.1285, 30.2979],
                max_pixel_value=255.0
            ),
            A.Transpose(p=1),
            ToTensorV2(),
        ]
    )
    method = ['original', 'HorizontalFlip', 'VerticalFlip', 'Transpose']
    transform_complicated = [test_transform, test_complicated_transfom1, test_complicated_transfom2,
                             test_complicated_transfom3]
    transform_dict = dict(zip(method, transform_complicated))

    test(model, TEST_DATA_DIR, test_transform, device=DEVICE)
    # complicated_test(model, TEST_DATA_DIR, transform_dict=transform_dict,device=DEVICE)


