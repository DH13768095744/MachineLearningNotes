# torch加载图片

大部分时候，图片太多，但内存无法一次性载入这么多张图片，所以需要分批次载入（这和batch size不一样，batch size既是为了解决显存问题，也是为了不要陷入局部最优）在PyTorch中，可以使用`torch.utils.data.Dataset`和`torch.utils.data.DataLoader`来加载和处理大量的图像数据

~~~python
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 假设你有一个包含所有图片路径的列表
image_paths = [...]  # 包含所有图片路径的列表

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 添加任何其他必要的转换
])

# 创建数据集实例
dataset = CustomDataset(image_paths, transform=transform)

# 创建数据加载器
batch_size = 128
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 在你的训练循环中使用数据加载器
for batch in data_loader:
    # batch 是一个张量，大小为 [batch_size, channels, height, width]
    # 在这里编写你的训练代码
    pass

~~~















