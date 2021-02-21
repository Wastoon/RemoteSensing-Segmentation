import os
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

import torch
import torch.nn as nn
from tqdm import tqdm
import glob
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

import numpy as np
from utils import load_checkpoint
import cv2
from model import get_model
from data_process import image_open
from train_v2 import check_path

###Hyperparameter 
TEST_DATA_DIR = 'data/origins/suichang_round1_test_partA_210120/suichang_round1_test_partA_210120'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

MODEL_DIR = '3channel_unet_v1'
MODEL_NAME = '21epoch.pth.tar'
CHECKPOINT = f'results/{MODEL_DIR}/{MODEL_NAME}'
SAVE_PATH = f'{MODEL_DIR}_{MODEL_NAME}_/'
CHANNEL_NUM = 3
ENCODER='resnext101_32x8d'
ENCODER_WEIGHT='swsl'


def test(model, test_data_dir, transform=None,device='cuda', data_channel_num=3):
    model.eval()
    
    for idx, name in enumerate(tqdm(glob.glob(os.path.join(test_data_dir, '*.tif'))[:])):
        image, Nir, NDVI, NDWI, IRRG, SAVI, Grass = image_open(name)
        # masks = {'image':image,
        #          'Nir':Nir,
        #          'NDVI':NDVI, 
        #          'NDWI':NDWI, 
        #          'IRRG':IRRG,
        #          'SAVI':SAVI,
        #          'Grass':Grass}

        masks = {'image':image,
            'Nir':Nir}

        if transform is not None:
            augmentations = transform(**masks)
            image = augmentations['image']
            Nir = torch.from_numpy(augmentations['Nir']).unsqueeze(0)
            # NDVI = torch.from_numpy(augmentations['NDVI']).unsqueeze(0)
            # NDWI = torch.from_numpy(augmentations['NDWI']).unsqueeze(0)
            # IRRG = torch.from_numpy(augmentations['IRRG']).unsqueeze(0)
            # SAVI = torch.from_numpy(augmentations['SAVI']).unsqueeze(0)
            # Grass = torch.from_numpy(augmentations['Grass']).unsqueeze(0)
        NDVI = torch.from_numpy(NDVI).unsqueeze(0)
        NDWI = torch.from_numpy(NDWI).unsqueeze(0)
        IRRG = torch.from_numpy(IRRG).unsqueeze(0)
        SAVI = torch.from_numpy(SAVI).unsqueeze(0)
        Grass = torch.from_numpy(Grass).unsqueeze(0)

        image = torch.cat([image, Nir, NDVI, NDWI, IRRG, SAVI, Grass], dim=0)


        with torch.no_grad():
            image = image[:data_channel_num,:,:].unsqueeze(0).to(device=device)
            score = model(image).clone().cpu().softmax(1).numpy()

            score_sigmoid = score[0].argmax(0) + 1
            assert score_sigmoid.max() <=10, "category pred is wrong, bigger than 10"
            assert  score_sigmoid.min() >=1, "category pred is wrong, smaller than 1"

            cv2.imwrite(SAVE_PATH + os.path.basename(name).replace('.tif', '.png'), score_sigmoid)

def test_batch(model, test_data_dir, transform=None,device='cuda', data_channel_num=3, batch_size=24):
    model.eval()
    all_img_list = glob.glob(os.path.join(test_data_dir, '*.tif'))[:]
    length_batch = len(all_img_list) / batch_size

    img_list = []
    name_list = []
    for idx, name in enumerate(tqdm(glob.glob(os.path.join(test_data_dir, '*.tif'))[:])):
        image, Nir, NDVI, NDWI, IRRG, SAVI, Grass = image_open(name)
        masks = {'image':image,
                 'Nir':Nir,
                 'NDVI':NDVI, 
                 'NDWI':NDWI, 
                 'IRRG':IRRG,
                 'SAVI':SAVI,
                 'Grass':Grass}

        # masks = {'image':image,
        #     'Nir':Nir}

        if transform is not None:
            augmentations = transform(**masks)
            image = augmentations['image']
            Nir = torch.from_numpy(augmentations['Nir']).unsqueeze(0)
            NDVI = torch.from_numpy(augmentations['NDVI']).unsqueeze(0)
            NDWI = torch.from_numpy(augmentations['NDWI']).unsqueeze(0)
            IRRG = torch.from_numpy(augmentations['IRRG']).unsqueeze(0)
            SAVI = torch.from_numpy(augmentations['SAVI']).unsqueeze(0)
            Grass = torch.from_numpy(augmentations['Grass']).unsqueeze(0)
        # NDVI = torch.from_numpy(NDVI).unsqueeze(0)
        # NDWI = torch.from_numpy(NDWI).unsqueeze(0)
        # IRRG = torch.from_numpy(IRRG).unsqueeze(0)
        # SAVI = torch.from_numpy(SAVI).unsqueeze(0)
        # Grass = torch.from_numpy(Grass).unsqueeze(0)

        image = torch.cat([image, Nir, NDVI, NDWI, IRRG, SAVI, Grass], dim=0).unsqueeze(0)

        img_list.append(image)
        name_list.append(name)
        if (idx+1)%batch_size == 0:
            image_batch = torch.cat(img_list, dim=0)
            with torch.no_grad():
                image_batch = image_batch[:,:data_channel_num,:,:].to(device=DEVICE)
                score = model(image_batch).clone().cpu().softmax(1).numpy()
                score_sigmoid = score.argmax(1) + 1

                for res_id, peer_name in enumerate(name_list):
                    cv2.imwrite(SAVE_PATH+ os.path.basename(peer_name).replace('.tif', '.png'), score_sigmoid[res_id])

            img_list = []
            name_list = []

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
                        score = score.transpose((0, 2,1))


                    score_list.append(score)

            for score in score_list:
                res = res + score
            res = res / len(score_list)
            score_sigmoid = res.argmax(0) + 1
            assert score_sigmoid.max() <= 10, "category pred is wrong, bigger than 10"
            assert score_sigmoid.min() >= 1, "category pred is wrong, smaller than 1"

            cv2.imwrite(SAVE_PATH + os.path.basename(name).replace('.tif', '.png'), score_sigmoid)

if __name__=="__main__":

    check_path(SAVE_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    model = get_model(data_channel=CHANNEL_NUM,
                      encoder=ENCODER,
                      encoder_weight=ENCODER_WEIGHT).to(device=DEVICE)
    
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

    method = ['original','HorizontalFlip', 'VerticalFlip', 'Transpose']

    #test(model, TEST_DATA_DIR, test_transform, device=DEVICE, data_channel_num=CHANNEL_NUM)

    test_batch(model, TEST_DATA_DIR, test_transform, device=DEVICE, data_channel_num=CHANNEL_NUM, batch_size=24)



