from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import nibabel as nib
import os
from google.colab import drive
import h5py

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = self.pool(F.relu(self.conv2(x1)))
        x3 = self.upconv(x2)
        return torch.sigmoid(self.final(x3))

class BraTSDataset(Dataset):
    def __init__(self,image_dir):
        self.image_dir = image_dir
        self.images = os.listdir(self.image_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        img_path = os.path.join(self.image_dir,self.images[idx])
        image_nifti = nib.load(img_path).get_fdata()

        image = np.array(image_nifti,dtype=np.float32)

        image = (image-np.min(image))/(np.max(image) - np.min(image))

        image = torch.tensor(image, dtype=torch.float32)

        return image

base_path = '/content/drive/My Drive/3/BraTS2020_training_data'
train_image_path = os.path.join(base_path,"content")

train_dataset = BraTSDataset(train_image_path,)
train_loader = DataLoader(train_dataset,batch_size=4,shuffle =True)

Model = UNet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
  for images in train_loader:
    optimizer.zero_grad()
    output = Model(images)
    loss = criterion(output)
    loss.backward()
    optimizer.step()
  print(f"epochs {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def dice_coefficient(pred,target,epsilon=1e-6):
  intersection = (pred*target).sum()
  return (2.0*intersection+epsilon)/(pred.sum()+target.sum()+epsilon)

Model.eval()
dice_scores = []
with torch.no_grad():
  for images in train_loader:
    outputs = Model(images)
    loss = criterion(outputs)
    test_loss += loss.item()
    dice_scores.append(dice_score.item())

mean_dice = np.mean(dice_scores)
print(f"Mean Dice Score: {mean_dice:.4f}")

