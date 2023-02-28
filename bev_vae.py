import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader


class CarlaDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        f = h5py.File("carla_dataset.hdf5", "r") 
        bev_image = f['bev_image'][()]
        bev_image = (bev_image / 255).astype('float32')
        self.len = bev_image.shape[0]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=64),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.x_data = bev_image
        self.y_data = 0

    def __getitem__(self, index):
        x = self.transform(self.x_data[index]) # C*H*W 
        return x , 0.0

    def __len__(self):
        return self.len



# 设置设备
torch.cuda.set_device(0)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# 如果没有文件夹就创建一个文件夹
sample_dir = 'samples_vae'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

image_size = 64
z_dim = 32
num_epochs = 200
batch_size = 128
learning_rate = 1e-5 
PATH = './vae_net.pth'
continue_train = True

# 加载同时做transform预处理
dataset = CarlaDataset()

data_loader = DataLoader( dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)
    
class VAE(nn.Module):
    def __init__(self,  image_channels=3, h_dim=1024, z_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        self.fc3 = nn.Linear(z_dim, h_dim)
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        x_reconst = self.decoder(z)
        return x_reconst

    def forward(self, x):
        # print(f'x = {x.shape}') (128, 3, 64, 64)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        # print(f'x_reconst = {x_reconst.shape}') (128, 3, 64, 64)

        return x_reconst, mu, log_var


model = VAE().to(device)

if continue_train:
    model.load_state_dict(torch.load(PATH))

optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.1)

try:

    for epoch in range(num_epochs):
        for i,(x,_) in enumerate(data_loader):
            x = x.to(device)
            x_reconst,mu,log_var = model(x)
            reconst_loss = F.binary_cross_entropy(x_reconst,x,reduction='sum')

            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            loss = reconst_loss + kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (i+1) % 100 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                    .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))

        with torch.no_grad():
            # 随机生成的图像
            z = torch.randn(batch_size, z_dim).to(device)
            out = model.decode(z).view(-1, 3, 64, 64)
            save_image(out,os.path.join(sample_dir,'sample-{}.png'.format(epoch+1)))
            
            # 重构的图像
            out, _, _ = model(x)
            x_concat = torch.cat([x, out], dim=3)
            save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))

finally:
    torch.save(model.state_dict(),PATH)

