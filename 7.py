import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode, Resize, CenterCrop, ToTensor, Normalize
import tifffile
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = []
        cls_idx = 0
        for class_name in self.classes:
            cls_idx = cls_idx + 1
            class_dir = os.path.join(self.root_dir, class_name)
            rgb_images = sorted(os.listdir(os.path.join(class_dir, 'RGB')))
            ms_images = sorted(os.listdir(os.path.join(class_dir, 'MS')))
            for rgb_image in rgb_images:
                ms_image_names = random.sample(ms_images, 3)
                for ms_image in ms_image_names:
                    rgb_path = os.path.join(class_dir, 'RGB', rgb_image)
                    ms_path = os.path.join(class_dir, 'MS', ms_image)
                    with open(rgb_path, 'rb') as rgb_file, open(ms_path, 'rb') as ms_file:
                        # RGB Image
                        BICUBIC = InterpolationMode.BICUBIC
                        rgb_img = Image.open(rgb_file)
                        trans = transforms.Compose([Resize(256, interpolation=BICUBIC), CenterCrop(256),
                                                    ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                          (0.26862954, 0.26130258, 0.27577711))])
                        rgb_img = trans(rgb_img)

                        # MS Image
                        ms_img = (tifffile.imread(ms_file) / 256).astype(np.uint8)
                        ms_img = torch.tensor(ms_img, dtype=torch.int64)
                        ms_img = np.transpose(ms_img, (2, 0, 1))

                        # Pairing
                        pair = (rgb_img, ms_img, cls_idx)
                        pairs.append(pair)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# Initialize dataset and dataloaders
st = time.time()
root_dir = r'C:\Users\yuvar\Desktop\Data_files - Copy\CTrain'
train_dataset = MyDataset(root_dir)

root_dir = r'C:\Users\yuvar\Desktop\Data_files - Copy\CVal'
val_dataset = MyDataset(root_dir)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
ed = time.time()
print(f'Time to load data: {ed - st}')


# Define Network1
class Network1(nn.Module):
    def __init__(self, in_channels):
        super(Network1, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(7, 7), stride=(3, 3), padding=(0, 0))
        self.pool1 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0))
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.pool4 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.linear1 = nn.Linear(in_features=256, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=1024)
        self.linear3 = nn.Linear(in_features=1024, out_features=16)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        return x

x=torch.randn(64,1,256,256)
in_channels = 1  # Extracting the number of input channels from the tensor x
model=Network1(in_channels)
print(model(x).shape)

# Define Network2
class Network2(nn.Module):
    def __init__(self, in_channels):
        super(Network2, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0))
        self.pool1 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.pool2 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.pool3 = nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.linear1 = nn.Linear(in_features=256, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=1024)
        self.linear3 = nn.Linear(in_features=1024, out_features=16)

    def forward(self, y):
        y = self.relu(self.conv1(y))
        y = self.pool1(y)
        y = self.relu(self.conv2(y))
        y = self.pool2(y)
        y = self.relu(self.conv3(y))
        y = self.pool3(y)
        y = y.reshape(y.shape[0], -1)
        y = self.relu(self.linear1(y))
        y = self.linear2(y)
        y = self.linear3(y)
        return y

y=torch.randn(64,1,64,64)
in_channels = 1  # Extracting the number of input channels from the tensor x
model=Network2(in_channels)
# Pass the reshaped y to the model
print(model(y).shape)

class CombinedNetwork(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_ms):
        super(CombinedNetwork, self).__init__()
        self.network1 = Network1(in_channels=in_channels_rgb)
        self.network2 = Network2(in_channels=in_channels_ms)

    @property
    def dtype_M1(self):
        return self.network1.conv1.weight.dtype

    @property
    def dtype_M2(self):
        return self.network2.conv1.weight.dtype

    def encode_M1(self, image):
        return self.network1(image.type(self.dtype_M1))

    def encode_M2(self, image):
        return self.network2(image.type(self.dtype_M2))

    def forward(self, rgb_input, ms_input):
        output1 = self.encode_M1(rgb_input)
        output2 = self.encode_M2(ms_input)
        return output1, output2


# Custom loss function
class CrossFunctionsLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(CrossFunctionsLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, SU, SP, SM, FP, FM, B):
        loss_IRSC = self.compute_IRSC(SU, FP, FM)
        loss_IASC = self.compute_IASC(SP, SM, FP, FM)
        loss_BQC = self.compute_BQC(FP, FM, B)
        loss_FDC = self.compute_FDC(FP, FM)

        total_loss = loss_IRSC + self.alpha * loss_IASC  + self.beta * loss_BQC + self.gamma * loss_FDC
        #print("IRSC:", loss_IRSC)
        #print("IASC:", loss_IASC)
        #print("BQC:", loss_BQC)
        #print("FDC:", loss_FDC)
        return total_loss

    def compute_IRSC(self, SU, FP, FM):
        Omega_U = FP.t().mm(FM)/2

        epsilon = 1e-8
        loss_IRSC=0
        for i in range(Omega_U.shape[0]):
            for j in range(Omega_U.shape[1]):
                loss_IRSC += ((-SU[i,j,0] * Omega_U[i,j]) + torch.log(1 + Omega_U[i,j]))
        if torch.isnan(loss_IRSC):
            print("IRSC is NaN")
            print(SU[:, :, 0])
            print(Omega_U)
        return loss_IRSC

    def compute_IASC(self, SP, SM, FP, FM):
        Omega_P = FP.t().mm(FP)/2
        Omega_M = FM.t().mm(FM)/2
        epsilon = 1e-8
        loss_SP=0
        loss_SM=0
        for i in range(Omega_P.shape[0]):
            for j in range(Omega_P.shape[1]):
                loss_SP += (-SP[i, j, 0] * Omega_P[i, j] + torch.log(1 + Omega_P[i, j]+ epsilon))
                loss_SM += (-SM[i, j, 0] * Omega_M[i, j] + torch.log(1 + Omega_M[i, j]+ epsilon))



        loss_IASC = loss_SP + loss_SM
        return loss_IASC

    def compute_BQC(self, FP, FM, B):
        loss_BQC = torch.norm(FP - B, 'fro')  + torch.norm(FM - B, 'fro')
        return loss_BQC

    def compute_FDC(self, FP, FM):
        ones_vector = torch.ones(FP.size(1), 1, device=FP.device)
        loss_FDC = torch.norm(FP.mm(ones_vector), 'fro') ** 2 + torch.norm(FM.mm(ones_vector), 'fro') ** 2
        return loss_FDC

# Function to calculate accuracy
def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy


def compute_similarity_matrices(targets_rgb, targets_ms):
    batch_size = targets_rgb.size(0)
    SU = torch.zeros(batch_size, batch_size, 2, device=targets_rgb.device)
    SP = torch.zeros(batch_size, batch_size, 2, device=targets_rgb.device)
    SM = torch.zeros(batch_size, batch_size, 2, device=targets_ms.device)

    for i in range(batch_size):
        for j in range(batch_size):
            # SU matrix
            if targets_rgb[i] == targets_ms[j]:
                SU[i, j] = torch.tensor([1, 0], device=targets_rgb.device)
            else:
                SU[i, j] = torch.tensor([0, 1], device=targets_rgb.device)

            # SP matrix
            if targets_rgb[i] == targets_rgb[j]:
                SP[i, j] = torch.tensor([1, 0], device=targets_rgb.device)
            else:
                SP[i, j] = torch.tensor([0, 1], device=targets_rgb.device)

            # SM matrix
            if targets_ms[i] == targets_ms[j]:
                SM[i, j] = torch.tensor([1, 0], device=targets_ms.device)
            else:
                SM[i, j] = torch.tensor([0, 1], device=targets_ms.device)

    return SU, SP, SM

# Validation function
def validate_model(combined_model, val_loader, loss_fn, device='cuda'):
    combined_model.eval()
    val_loss = 0.0
    val_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, (data1, data2, targets) in enumerate(val_loader):
            inputs_rgb = data1.to(device)
            inputs_ms = data2.to(device)
            targets = targets.to(device)

            outputs1, outputs2 = combined_model(inputs_rgb, inputs_ms)

            SU, SP, SM = compute_similarity_matrices(targets, targets)

            FP = outputs1
            FM = outputs2

            B = FP + FM
            B[B > 0] = 1
            B[B < 0] = -1

            loss = loss_fn(SU, SP, SM, FP.t(), FM.t(), B.t())

            val_loss += loss.item()
            val_accuracy += calculate_accuracy(FP, targets)

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)

    return val_loss, val_accuracy

# Training function
def train_model(combined_model, train_loader, loss_fn, optimizer, num_epochs=1, device='cuda'):
    combined_model.train()
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0


        for batch_idx, (data1, data2, targets) in enumerate(train_loader):
            inputs_rgb = data1.to(device)
            inputs_ms = data2.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()


            outputs1, outputs2 = combined_model(inputs_rgb, inputs_ms)

            # Example of defining SU, SP, SM, FP, FM, B (replace with actual computation)
            SU, SP, SM = compute_similarity_matrices(targets, targets)


            FP = outputs1
            FM = outputs2
            #FP = normalize(FP, p=2.0, dim=1)
            #FM = normalize(FM, p=2.0, dim=1)

            B = FP+FM
            B[B > 0] = 1
            B[B < 0] = -1

            loss = loss_fn(SU, SP, SM, FP.t(), FM.t(), B.t())
            loss.backward()


            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(combined_model.parameters(), max_norm=1.0)


            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += calculate_accuracy(FP, targets)

        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)

        val_loss, val_accuracy = validate_model(combined_model, val_loader, loss_fn, device)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(combined_model.state_dict(), 'best_combined_model.pth')


# Initialize models, loss function, and optimizers
device = 'cuda' if torch.cuda.is_available() else 'cpu'
in_channels_rgb = 3  # Number of input channels for RGB
in_channels_ms = 13  # Number of input channels for MS
combined_model = CombinedNetwork(in_channels_rgb, in_channels_ms).to(device)

alpha, beta, gamma = 1.0, 0.1, 1.0
loss_fn = CrossFunctionsLoss(alpha, beta, gamma)

optimizer = optim.SGD(combined_model.parameters(), lr=0.001)

# Train the models
train_model(combined_model, train_loader, loss_fn, optimizer, num_epochs=10, device=device)
