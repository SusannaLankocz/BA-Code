import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "datasets/train"
VAL_DIR = "datasets/test"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 150
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_A = "output/netG_A2P.pth"
CHECKPOINT_GEN_P = "output/netG_P2A.pth"
CHECKPOINT_DISC_A = "output/netD_A.pth"
CHECKPOINT_DISC_P = "output/netD_P.pth"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.RandomCrop(256, 256),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.1),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)