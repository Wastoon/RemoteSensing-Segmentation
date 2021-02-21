from torch.utils.data import Dataset, DataLoader
import albumentations as A
from PIL import Image
import numpy as np
import os
from albumentations.pytorch import ToTensorV2
from data_process import image_open
import torch
from tqdm import tqdm


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = (channels_sum / num_batches)
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


class RemoteSensingMap(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, test_mode=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)
        self.test_mode = test_mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.tif', '.png'))
        image, NIR = np.array(Image.open(img_path), dtype=np.float32)[:, :, :-1], np.array(Image.open(img_path),
                                                                                           dtype=np.float32)[:, :, -1]
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.long) - 1

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image, mask = augmentations['image'], augmentations['mask']

        return image, mask


class RemoteSensingMap_aug(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, test_mode=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)
        self.test_mode = test_mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.tif', '.png'))
        image, Nir, NDVI, NDWI, IRRG, SAVI, Grass = image_open(img_path)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.long) - 1
        masks = {'image': image,
                 'mask': mask,
                 'Nir': Nir,
                 'NDVI': NDVI,
                 'NDWI': NDWI,
                 'IRRG': IRRG,
                 'SAVI': SAVI,
                 'Grass': Grass}

        if self.transform is not None:
            augmentations = self.transform(**masks)
            image, mask = augmentations['image'], augmentations['mask']
            Nir = torch.from_numpy(augmentations['Nir']).unsqueeze(0)
            NDVI = torch.from_numpy(augmentations['NDVI']).unsqueeze(0)
            NDWI = torch.from_numpy(augmentations['NDWI']).unsqueeze(0)
            IRRG = torch.from_numpy(augmentations['IRRG']).unsqueeze(0)
            SAVI = torch.from_numpy(augmentations['SAVI']).unsqueeze(0)
            Grass = torch.from_numpy(augmentations['Grass']).unsqueeze(0)

        image = torch.cat([image, Nir, NDVI, NDWI, IRRG, SAVI, Grass], dim=0)

        return image, mask


def test():
    image_dir, mask_dir = 'data/train_imgs', 'data/train_masks'

    test_transform = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.Normalize(
                mean=[0.625, 0.448, 0.688],
                std=[0.131, 0.177, 0.101],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    test_dataset = RemoteSensingMap(image_dir, mask_dir, test_transform)
    test_dataset = RemoteSensingMap_aug(image_dir, mask_dir, test_transform)

    img, mask = test_dataset[100]

    print(len(test_dataset))
    print(img.shape)
    print(mask.shape)
    print(img.max())


def calculate_mean_std():
    image_dir, mask_dir = 'data/train_imgs', 'data/train_masks'
    test_transform = A.Compose(
        [
            ToTensorV2(),
        ]
    )
    test_dataset = RemoteSensingMap(image_dir, mask_dir, transform=test_transform)

    test_Dataloader = DataLoader(test_dataset, batch_size=24, shuffle=True)
    mean, std = get_mean_std(test_Dataloader)
    print(mean)
    print(std)


if __name__ == '__main__':
    test()
    # calculate_mean_std()
