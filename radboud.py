import argparse
import copy
import glob
import json
import logging
import os
import time
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets.vision import VisionDataset
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.transforms as transforms

from PIL import Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


data_channels = ['training']


class DriveDataset(VisionDataset):
    def __init__(self,
                 channel: str,
                 dataset_path: str = './data'
    ):
        self.image_path = os.path.join(dataset_path, channel, 'images')
        self.image_files = glob.glob(
            os.path.join(self.image_path, '*.tif'))

        self.mask_path = os.path.join(dataset_path, channel, '1st_manual')
        self.mask_files = [
            i.replace('/images/', '/1st_manual/').replace('_training.tif', '_manual1.gif') \
            for i in self.image_files
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((413, 400), 2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        #print(f'Retrieved item with index = {idx}')
        image_fname = self.image_files[idx]
        mask_fname = self.mask_files[idx]
        logger.debug(f"Fetching image file {image_fname} and mask file {mask_fname}")
        with open(image_fname, 'rb') as img_f, open(mask_fname, 'rb') as mask_f:
            X_image = Image.open(img_f)
        
            if self.transform:
                X_image = self.transform(X_image)

            transform_mask = transforms.Compose([
                transforms.Resize((413, 400), 2),
                transforms.ToTensor(),
            ])
            y_image = Image.open(mask_f).convert('L')
            y_image = transform_mask(y_image)
        
        return X_image, y_image


def _get_dataloaders(data_dir, batch_size, **kwargs):
    datasets = {
        channel: DriveDataset(channel=channel, dataset_path=data_dir) \
        for channel in data_channels
    }

    dataloaders = {
        channel: DataLoader(datasets[channel],
                            batch_size=batch_size,
                            shuffle=True,
                            **kwargs)
        for channel in data_channels
    }
    
    return dataloaders    
    

def _get_model():
    # DeepLab V3 encoder, ResNet 101 decoder, use pretrained model.
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    # Feature vector size = 2048 (resnet101), output channels = 1
    model.classifier = DeepLabHead(2048, 1)
    
    return model


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    model_path = os.path.join(model_dir, 'model.pth')
    # Recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), model_path)
    
    
def train(args):
    logger.debug(f"Number of GPUs available = {args.num_gpus}")
    
    model = _get_model()
    model.train()
    
    n_epochs = args.epochs
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    loss_function = torch.nn.MSELoss(reduction='mean')
    
    logger.debug(f'Learning rate = {args.lr}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Start metering the time it takes to train.
    start_time = time.time()
    
    field_names = ['epoch', 'train_loss', 'test_loss'] + \
        ['train_{m}' for m in metrics.keys()] + \
        ['test_{m}' for m in metrics.keys()]
    
    # Use GPU if available.
    device_id = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_id)
    
    # Send the model to the CPU/GPU.
    model.to(device)
    
    print(f'Device id = {device_id}')
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    logger.debug(f'Batch size = {args.batch_size}')
    dataloaders = _get_dataloaders(args.data_dir, args.batch_size, **kwargs)
    
    # Iterate for every epoch.
    for epoch in range(1, n_epochs + 1):
        logger.info(f'Epoch {epoch}/{n_epochs}')
        
        # Iterate for every data channel (training|test)
        for phase in data_channels:
            logger.debug(f'Entering {phase} phase.')
            if phase == 'training':
                model.train()
            else:
                model.eval()
            
            for inputs, mask in dataloaders[phase]:
                inputs = inputs.to(device)
                mask = mask.to(device)
                
                # Zero the parameter gradients.
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    loss = loss_function(outputs['out'], mask)
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = mask.data.cpu().numpy().ravel()
                    
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            metric(y_true > 0, y_pred > 0.1)
                        else:
                            metric(y_true.astype('uint8'), y_pred)
                    
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()
            epoch_loss = loss
            logger.info('{} loss {:.4f}' . format(phase, loss))

    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s' . format(
        time_elapsed // 60, time_elapsed % 60))
    
    _save_model(model, args.model_dir)
    
    
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _get_model()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=3, metavar='N',
                        help='input batch size for training (default: 3)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1E-4)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())
