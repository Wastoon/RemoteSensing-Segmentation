import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from model import get_model
import matplotlib.pyplot as plt
from utils import (
load_checkpoint,
save_checkpoint,
get_loaders,
check_valid_metric
)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import lovasz_softmax as L
import torch.nn.functional as F
##Hyperparameters
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 60
NUM_WORKS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'data/train_imgs'
TRAIN_MASK_DIR = 'data/train_masks'
FOLD_VAL = 5

TEST_IMG_DIR = 'data/test_imgs'


EXP = '3channel_unet_v1'
CHANNEL_NUM = 3
ENCODER='resnext101_32x8d'
ENCODER_WEIGHT='swsl'
BATCH_SIZE = 24

EXP = '4channel_unet_v1'
CHANNEL_NUM = 4
ENCODER='resnext101_32x8d'
ENCODER_WEIGHT='swsl'
BATCH_SIZE = 24

EXP = '3channel_efficientb7_v1'
CHANNEL_NUM = 3
ENCODER='timm-efficientnet-b7'
ENCODER_WEIGHT='noisy-student'
BATCH_SIZE = 16


PLOT_PATH = f'plot_img_{EXP}.png'
SAVE_PATH = f'/data3/mry/results/{EXP}/'


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        return path


def train_fn(loader, model, optimizer, scheduler, loss_fn, scaler, epoch, aux_loss=None):
    loop = tqdm(loader)
    losses = []
    loss_dict = {}

    for batch_idx, (data, target) in enumerate(loop):
        data = data[:,:CHANNEL_NUM,:,:].to(device=DEVICE)
        #print(data.shape)
        target = target.long().to(device=DEVICE)

        ##forward
        with torch.cuda.amp.autocast():
            predictiions = model(data)
            loss = loss_fn(predictiions, target)
            loss_dict['SoftLoss'] = loss.item()
            if aux_loss:
                predictiions = F.softmax(predictiions, dim=1)
                aux_loss = L.lovasz_softmax(predictiions, target)
                loss_dict['auxLoss'] = aux_loss .item()
                loss = loss + 0.5 * aux_loss
                loss_dict['TotalLoss'] = loss.item()

        ##backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        ##update tqdm loop
        #loss=loss.item(), aux_loss=aux_loss.item()
        if not aux_loss:
            loop.set_postfix(loss=loss.item())
        else:
            loop.set_postfix(
                Softloss = loss_dict['SoftLoss'],
                auxLoss = loss_dict['auxLoss'],
                TotalLoss = loss_dict['TotalLoss']
            )

        losses.append(loss.item())
        loop.set_description(f"[epoch][{epoch}/{NUM_EPOCHS}][Iteration][{batch_idx}/{len(loader.dataset)//BATCH_SIZE}]")

    return sum(losses) / len(losses), current_lr


def main():
    check_path(SAVE_PATH)
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.OneOf(
                [ 
                    A.Rotate(limit=35, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.5)
                ]
            ),
            A.Normalize(
                mean=[0.625, 0.448, 0.688],
                std=[0.131, 0.177, 0.101],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
                     
    model = get_model(data_channel=CHANNEL_NUM,
                      encoder=ENCODER,
                      encoder_weight=ENCODER_WEIGHT).to(device=DEVICE)
    print(model)
    loss_fn = nn.CrossEntropyLoss().to(device=DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    ##plot
    plot_train_loss = []
    plot_val_loss = []
    plot_dice = []
    plot_miou = []
    learning_lr = []

    # Define Scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1, patience=10, 
    #                                                   verbose=True)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  cycle_momentum=False,
                                                  base_lr=1.25e-4,
                                                  max_lr=0.001,
                                                  step_size_up=2000,
                                                  mode="triangular2", 
                                                  verbose=False)
    best_iou = 0
    best_dice = 0
    

    train_loader, val_loader = get_loaders(train_dir=TRAIN_IMG_DIR,
                                            train_maskdir=TRAIN_MASK_DIR,
                                            batch_size=BATCH_SIZE,
                                            train_transform=train_transform,
                                            num_workers=NUM_WORKS,
                                            pin_memory=PIN_MEMORY)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        epoch_loss, current_lr = train_fn(train_loader, model, optimizer, scheduler, loss_fn, scaler, epoch, aux_loss='lovasz_softmax')
        plot_train_loss.append(epoch_loss)
        print(epoch_loss)
        learning_lr.append(current_lr)


        #save_checkpoint(check_point, filename=f"/data3/mry/results/best_checkpoint_{flod_idx}fold_{epoch}epoch.pth.tar")
        ##check valid metric
        m_dice, miou, val_loss = check_valid_metric(val_loader, model, device=DEVICE, loss_fn=loss_fn, aux_loss='lovasz_softmax', channel_nums=CHANNEL_NUM)
        plot_val_loss.append(val_loss if val_loss<100 else 100)
        plot_dice.append(m_dice)
        plot_miou.append(miou)

        if best_iou < miou:
            best_iou = miou
            ##save model
            check_point = {
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()
            }
            save_checkpoint(check_point, filename=f"{SAVE_PATH}{epoch}epoch.pth.tar")

        
        ##plot metric and save
        fig = plt.figure(figsize=(24, 12))
        x = [i for i in range(epoch+1)]

        ax = fig.add_subplot(2, 3, 1)
        ax.plot(x, plot_train_loss, label='train loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('train loss')
        ax.grid(True)
        ax = fig.add_subplot(2, 3, 2)
        ax.plot(x, plot_val_loss, label='val loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('val loss')
        ax.grid(True)

        ax = fig.add_subplot(2, 3, 3)
        ax.plot(x, learning_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.grid(True)

        ax = fig.add_subplot(2, 3, 4)
        ax.plot(x, plot_miou,  label='mIOU')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mIOU')
        ax.grid(True)

        ax = fig.add_subplot(2, 3, 5)
        ax.plot(x, plot_dice,  label='mDICE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mDICE')
        ax.grid(True)

        fig.savefig(PLOT_PATH)
        plt.show()

def test_loop(optimizer, scheduler):
    Epoch = 60
    Iteration = 534

    plot_lr = []
    lr = 0
    fig = plt.figure()
    for epoch in range(Epoch):
        x = [i for i in range(epoch+1)]
        for idx, it in enumerate(range(Iteration)):
            lr = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step(epoch+it/Iteration)
        plot_lr.append(lr)

    ax = fig.add_subplot(1,1,1)
    ax.plot(x, plot_lr,  label='Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.grid(True)
    fig.savefig('test_lr.png')


def test_schedule():
    optimizer = torch.optim.Adam(get_model(data_channel=CHANNEL_NUM,
                      encoder=ENCODER,
                      encoder_weight=ENCODER_WEIGHT).parameters(), lr=1e-3)
    
    scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer=optimizer,
                cycle_momentum=False,
                base_lr=1.25e-4,
                max_lr=0.001,
                   step_size_up=2000,
                    mode="triangular2", 
                    verbose=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)
    
    test_loop(optimizer, scheduler)

    





if __name__=='__main__':
    main()
    #test_schedule()
    


##TODO
#1. nir channel modify to make a 4-channel input  ---->Done!
#2. dic loss
#3. calculate mean and std ----->Done!
#4. establish test pipeline and fetch results ----->Done!
#5. scheduler optimizer   ----->Done! 


