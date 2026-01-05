import numpy as np
import torch
import cv2
import models
from torchvision import transforms
from PIL import Image
import os
import argparse
import logging
import os
from utils.helpers import get_instance, seed_torch
from torchvision.transforms import Grayscale, Normalize, ToTensor
from ruamel.yaml import safe_load, YAML
import torch.nn as nn
def normalization(imgs):

    mean = torch.mean(imgs)
    std = torch.std(imgs)
    n = Normalize([mean], [std])(imgs)
    n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))

    return n

# 预处理
transform = transforms.ToTensor()

yaml = YAML(typ='safe', pure=True)
with open('config.yaml', encoding='utf-8') as file:
       # CFG = Bunch(safe_load(file))
    CFG = yaml.load(file)
# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = get_instance(models, 'model', CFG)
net = nn.DataParallel(net.to('cpu'))
net.to(device)



net.load_state_dict(torch.load('checkpoints/checkpoint_epoch30.pth', map_location=device))
#state_dict = torch.load('pretrained_weights/loss/diceloss/checkpoint-epoch32.pth', map_location=torch.device('cpu'))

#net.load_state_dict(state_dict['state_dict'])


# 测试模式
net.eval()
# 读取所有图片路径
tests_path = os.listdir('predict/other/')  # 获取 './predict/' 路径下所有文件,这里的路径只是里面文件的路径

with torch.no_grad():  # 预测的时候不需要计算梯度
    for test_path in tests_path:  # 遍历每个predict的文件
        save_pre_path = 'predict/result_other/' + test_path.split('.')[-2] + '_res.tif'
        save_pre_path1 = 'predict/result_other/' + test_path.split('.')[-2] + '_r.tif' # 将保存的路径按照原图像的后缀，按照数字排序保存
        img = Image.open('predict/other/' + test_path)  # 预测图片的路径
        width, height = img.size[0], img.size[1]  # 保存图像的大小

        img = transform(img)
        data = data.to(device)

        img=normalization(img)

        img = torch.unsqueeze(img, dim=0)  # 扩展图像的维度

        pred = net(img)  # 网络预测

        pred = torch.squeeze(pred)  # 将(batch、channel)维度去掉print(pred)
        #print(pred)
       # pred = pred.argmax(dim=0)

        #predict = torch.sigmoid(pred).cpu().detach().numpy()
        predict = torch.sigmoid(pred)
        predict_b = np.where(predict >= 0.9, 1, 0)
     #   cv2.imwrite(save_pre_path1, np.uint8(predict * 255))
        cv2.imwrite(save_pre_path, np.uint8(predict_b*255))
        print(test_path)

