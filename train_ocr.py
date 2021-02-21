import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from model import get_model
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_valid_metric,
    check_valid_metric_ocr
)
from hr_ocr_model import CrossEntropy, OhemCrossEntropy, get_seg_model, main_OCR
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

##Hyperparameters
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_WORKS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'data/train_imgs'
TRAIN_MASK_DIR = 'data/train_masks'
FOLD_VAL = 5

TEST_IMG_DIR = 'data/test_imgs'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    losses = []
    for batch_idx, (data, target) in enumerate(loop):
        data = data[:, :3, :, :].to(device=DEVICE)
        # print(data.shape)
        target = target.long().to(device=DEVICE)

        ##forward
        with torch.cuda.amp.autocast():
            predictiions = model(data)

            loss = loss_fn(predictiions, target)
            loss = loss.mean()

        ##backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ##update tqdm loop
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())

    return sum(losses) / len(losses)


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.625, 0.448, 0.688],
                std=[0.131, 0.177, 0.101],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
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

    model, config = main_OCR()
    model.to(device=DEVICE)
    loss_fn = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                           weight=None, config=config).to(device=DEVICE)
    # optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    ##optimizer : only suppurt SGD
    params_dict = dict(model.named_parameters())
    if config.TRAIN.NONBACKBONE_KEYWORDS:
        bb_lr = []
        nbb_lr = []
        nbb_keys = set()
        for k, param in params_dict.items():
            if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                nbb_lr.append(param)
                nbb_keys.add(k)
            else:
                bb_lr.append(param)
        print(nbb_keys)
        params = [{'params': bb_lr, 'lr': config.TRAIN.LR},
                  {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}]
    else:
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

    optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )

    # Define Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10,
                                                           verbose=True)

    valid_idx, train_idx = [], []
    best_iou = 0
    best_dice = 0

    for flod_idx in range(FOLD_VAL):
        train_loader, val_loader = get_loaders(train_dir=TRAIN_IMG_DIR,
                                               train_maskdir=TRAIN_MASK_DIR,
                                               batch_size=BATCH_SIZE,
                                               train_transform=train_transform,
                                               num_workers=NUM_WORKS,
                                               pin_memory=PIN_MEMORY,
                                               fold_idx=FOLD_VAL,
                                               train_idx=train_idx,
                                               val_idx=valid_idx)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(NUM_EPOCHS):
            epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

            scheduler.step(epoch_loss)

            ##save model
            check_point = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(check_point,
                            filename=f"/data3/mry/results_OCR/best_checkpoint_{flod_idx}fold_{epoch}epoch.pth.tar")
            ##check valid metric
            m_dice, miou = check_valid_metric_ocr(val_loader, model, device=DEVICE, config=config)
            miou = miou
            if best_iou < miou or best_dice < m_dice:
                best_iou = miou
                best_dice = m_dice
                save_checkpoint(check_point,
                                filename=f"/data3/mry/results_01_adam/best_checkpoint_{flod_idx}fold_{epoch}epoch.pth.tar")
            ##print some examples to folder
            # TODO


if __name__ == '__main__':
    main()

##TODO
# 1. nir channel modify to make a 4-channel input  ---->Done!
# 2. dic loss
# 3. calculate mean and std ----->Done!
# 4. establish test pipeline and fetch results ----->Done!
# 5. scheduler optimizer   ----->Done!


