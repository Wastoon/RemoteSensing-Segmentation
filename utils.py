import os
import shutil
# import tqdm
import torch
import torchvision
from dataset import RemoteSensingMap, RemoteSensingMap_aug
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F


def split_dataset_to_different_folder(train_folder_path, new_folder_path_img, new_folder_path_mask):
    img_lists = os.listdir(train_folder_path)
    for idx, ele in enumerate(img_lists):
        img_path = os.path.join(train_folder_path, ele)
        if os.path.isfile(img_path):
            prefix = ele.split('.')[1]
            if prefix == 'tif':
                new_path = os.path.join(new_folder_path_img, ele)
            else:
                new_path = os.path.join(new_folder_path_mask, ele)
            shutil.copy(img_path, new_path)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_maskdir,
        batch_size,
        train_transform,
        num_workers=4,
        pin_memory=True,
        fold_idx=0,
        train_idx=None,
        val_idx=None
):
    # total_train_ds = RemoteSensingMap(
    ##    image_dir=train_dir,
    #    mask_dir=train_maskdir,
    #    transform=train_transform,
    # )
    total_train_ds = RemoteSensingMap_aug(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    for i in range(len(total_train_ds)):
        if i % fold_idx == 0:
            val_idx.append(i)
        else:
            train_idx.append(i)
    train_ds = torch.utils.data.Subset(total_train_ds, train_idx)
    val_ds = torch.utils.data.Subset(total_train_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def calculate_F1(pred, mask, c=10):
    iou_result = []
    for idx in range(c):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)

        uion = p.sum() + t.sum()
        overlap = (p * t).sum()
        iou = 2 * overlap / (p.sum() + t.sum() + 1e-5)
        iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result), np.stack(iou_result).mean()


def calculate_miou(pred, mask, c=10):
    iou_result = []
    for idx in range(c):
        t = (mask == idx).reshape(-1)
        p = (pred == idx).reshape(-1)

        overlap = t & p
        tp = overlap.sum()
        fp = (overlap ^ p).sum()
        fn = (overlap ^ t).sum()

        iou = (tp) / (fp + fn + tp + 1e-5)
        iou_result.append(iou.data.cpu().numpy())
    return np.stack(iou_result), np.stack(iou_result).mean()


def check_valid_metric(loader, model, device='cuda'):
    model.eval()
    val_dice = []
    val_iou = []

    with torch.no_grad():
        for x, y in loader:
            x = x[:, :3, :, :].to(device)
            y = y.to(device)
            preds = model(x)
            preds = preds.argmax(1)
            _, mean_dice = calculate_F1(preds, y)
            val_dice.append(mean_dice)
            _, mean_iou = calculate_miou(preds, y)
            val_iou.append(mean_iou)
    print(f"Got mean_dice_score:{sum(val_dice) / len(val_dice) * 100:.4f}")
    print(f"Got mean_iou:{sum(val_iou) / len(val_iou) * 100:.4f}")
    model.train()
    return sum(val_dice) / len(val_dice), sum(val_iou) / len(val_iou)


def check_valid_metric_ocr(loader, model, device='cuda', config=None):
    model.eval()
    val_dice_corase = []
    val_iou_croase = []
    val_dice_fine = []
    val_iou_fine = []

    with torch.no_grad():
        for x, y in loader:
            x = x[:, :3, :, :].to(device)
            y = y.to(device)
            pred_corase, pred_fine = model(x)

            ph, pw = pred_corase.size(2), pred_corase.size(3)
            h, w = y.size(1), y.size(2)
            if ph != h or pw != w:
                pred_corase = F.interpolate(input=pred_corase, size=(h, w),
                                            mode='bilinear',
                                            align_corners=config.MODEL.ALIGN_CORNERS)
                pred_fine = F.interpolate(input=pred_fine, size=(h, w),
                                          mode='bilinear',
                                          align_corners=config.MODEL.ALIGN_CORNERS)
            pred_corase = pred_corase.argmax(1)
            pred_fine = pred_fine.argmax(1)
            _, mean_dice = calculate_F1(pred_corase, y)
            val_dice_corase.append(mean_dice)
            _, mean_dice = calculate_F1(pred_fine, y)
            val_dice_fine.append(mean_dice)
            _, mean_iou = calculate_miou(pred_corase, y)
            val_iou_croase.append(mean_iou)
            _, mean_iou = calculate_miou(pred_fine, y)
            val_iou_fine.append(mean_iou)
        print(f"Got Croase mean_dice_score:{sum(val_dice_corase) / len(val_dice_corase) * 100:.4f}")
        print(f"Got Croase mean_iou:{sum(val_iou_croase) / len(val_iou_croase) * 100:.4f}")
        print(f"Got Croase mean_dice_score:{sum(val_dice_fine) / len(val_dice_fine) * 100:.4f}")
        print(f"Got Croase mean_iou:{sum(val_iou_fine) / len(val_iou_fine) * 100:.4f}")
    model.train()
    return sum(val_dice_fine) / len(val_dice_fine), sum(val_iou_fine) / len(val_iou_fine)


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()


def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


if __name__ == '__main__':
    train_folder_old = 'data/origins/suichang_round1_train_210120'
    new_imgs_folder = 'data/train_imgs'
    new_masks_folder = 'data/train_masks'

    split_dataset_to_different_folder(train_folder_old, new_imgs_folder, new_masks_folder)




