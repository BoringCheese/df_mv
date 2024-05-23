from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, r_images_path: list, v_images_path: list, images_class: list, transform=None):
        self.r_images_path = r_images_path
        self.v_images_path = v_images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.r_images_path)

    def __getitem__(self, item):
        r_img = Image.open(self.r_images_path[item])
        v_img = Image.open(self.v_images_path[item])
        # RGB为彩色图片，L为灰度图片
        if r_img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.r_images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            r_img = self.transform(r_img)
            v_img = self.transform(v_img)
        r_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        v_transform = transforms.Normalize([0.5], [0.5])
        r_img = r_transform(r_img)
        v_img = v_transform(v_img)
        return r_img, v_img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        r_images, v_images, labels = tuple(zip(*batch))

        r_images = torch.stack(r_images, dim=0)
        v_images = torch.stack(v_images, dim=0)
        labels = torch.as_tensor(labels)
        return r_images, v_images, labels
