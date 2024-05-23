import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.my_dataset import MyDataSet
from models.mobileVit.model import mobile_vit_xx_small as create_model
from models.mobileVit.model import NeuralNetwork
from utils.transforms import read_split_data, read_data
from utils.train_One_e import train_one_epoch, evaluate
from torch.utils.data import TensorDataset, DataLoader
from models.builder import EncoderDecoder as segmodel
from local_configs.NYUDepthv2.DFormer_Tiny import C

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def to_float_tensor(image):
    return image.float()


def main(args):
    args.config = 'local_configs.NYUDepthv2.DFormer_Tiny'
    config = C
    args.gpu = 1
    exec('from ' + args.config + ' import config')
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    tb_writer = SummaryWriter()
    train_path = os.path.join(args.rgb_data_path, "train")
    val_path = os.path.join(args.rgb_data_path, "val")
    r_train_images_path, train_images_label = read_data(train_path)
    val_images_path, val_images_label = read_data(val_path)
    v_train_images_path = [path.replace('\\flower_data\\', '\\flower_data_g\\') for path in r_train_images_path]
    v_val_images_path = [path.replace('\\flower_data\\', '\\flower_data_g\\') for path in val_images_path]
    img_size = 256

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()
                                     ]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor()
                                   ]),
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(r_images_path=r_train_images_path,
                              v_images_path=v_train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(r_images_path=val_images_path,
                            v_images_path=v_val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # train_dataset = ImageFolder('H:\\Img\\train', transform=data_transform["train"])
    # val_dataset = ImageFolder('H:\\Img\\val', transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               # collate_fn=train_dataset.collate_fn
                                               )
    # print(type(train_loader))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             # collate_fn=val_dataset.collate_fn
                                             )
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    BatchNorm2d = nn.BatchNorm2d
    model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d, single_GPU=1).to(device)
    # model = create_model(num_classes=args.num_classes).to(device)
    # print(model)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-2)

    best_acc = 0.930
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc, pred_0, la0 = train_one_epoch(model=model,
                                                             optimizer=optimizer,
                                                             data_loader=train_loader,
                                                             device=device,
                                                             epoch=epoch)

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        with open("H:\\data\\df_conv.txt", "a+") as f:
            print("[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, train_loss, train_acc), file=f)
            print("[val epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, val_loss, val_acc), file=f)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/best_model.pth")

        torch.save(model.state_dict(), "./weights/latest_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--gpus', help='used gpu number')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)

    # 数据集所在根目录  H:\\deep-learning-for-image-processing-master\\data_set\\flower_data\\flower_photos
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--rgb-data-path', type=str,
                        default="H:\\data_f\\flower_data")
    parser.add_argument('--vision-data-path', type=str,
                        default="H:\\Df_mbvit\\datasets\\flower_data\\vision")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
