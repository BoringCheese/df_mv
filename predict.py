import os
import json
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models.builder import EncoderDecoder as segmodel
import os
from local_configs.NYUDepthv2.DFormer_Tiny import C
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main():
    def to_float_tensor(image):
        return image.float()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = C
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         # transforms.Lambda(log_gabor1),
         transforms.ToTensor(),
         # transforms.Lambda(to_float_tensor),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "./img.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    BatchNorm2d = nn.BatchNorm2d
    model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d, single_GPU=1).to(device)
    # load model weights
    model_weight_path = "./weights/latest_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    end = time.perf_counter()
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    # plt.title(print_res)
    for i in range(5):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))


    print('Running time: %s Seconds' % (end - start))
    # plt.show()


if __name__ == '__main__':
    main()
