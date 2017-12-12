#coding:utf-8
import sys
import glob
sys.path.insert(1, "./crnn")
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import util
import dataset
from PIL import Image
import models.crnn as crnn
import keys
from math import *
# import mahotas
import cv2
import editdistance

def dumpRotateImage(img,degree,pt1,pt2,pt3,pt4):
    height,width=img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)
    
    
    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut=imgRotation[int(pt1[1]):int(pt3[1]),int(pt1[0]):int(pt3[0])]
    height,width=imgOut.shape[:2]
    return imgOut

def crnnSource(path):
    alphabet = keys.alphabet
    converter = util.strLabelConverter(alphabet)
    model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1)
    if torch.cuda.is_available():
        model = model.cuda()
    # path = './crnn/samples/netCRNN63.pth'
    # model.load_state_dict(torch.load(path))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=lambda storage, location: storage))
    
    return model,converter

def crnnRec(model,converter,im,text_recs):
   index = 0
   for rec in text_recs:
       pt1 = (rec[0],rec[1])
       pt2 = (rec[2],rec[3])
       pt3 = (rec[6],rec[7])
       pt4 = (rec[4],rec[5])
       partImg = dumpRotateImage(im,degrees(atan2(pt2[1]-pt1[1],pt2[0]-pt1[0])),pt1,pt2,pt3,pt4)
       #mahotas.imsave('%s.jpg'%index, partImg)
       

       image = Image.fromarray(partImg ).convert('L')
       #height,width,channel=partImg.shape[:3]
       #print(height,width,channel)
       #print(image.size) 

       #image = Image.open('./img/t4.jpg').convert('L')
       scale = image.size[1]*1.0 / 32
       w = image.size[0] / scale
       w = int(w)
       #print(w)

       transformer = dataset.resizeNormalize((w, 32))
       image = transformer(image)
       if torch.cuda.is_available():
           image = image.cuda()
       image = image.view(1, *image.size())
       image = Variable(image)
       model.eval()
       preds = model(image)
       _, preds = preds.max(2)
       preds = preds.squeeze(2)
       preds = preds.transpose(1, 0).contiguous().view(-1)
       preds_size = Variable(torch.IntTensor([preds.size(0)]))
       raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
       sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
       #print('%-20s => %-20s' % (raw_pred, sim_pred))
       print(index)
       print(sim_pred)
       index = index + 1


def crnn_line_rec(model,converter,image):
    scale = image.size[1]*1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    #print(w)

    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    # RuntimeError: dimension out of range (expected to be in range of [-2, 1], but got 2)
    # preds = preds.squeeze(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #print('%-20s => %-20s' % (raw_pred, sim_pred))
    # print(sim_pred)
    return sim_pred
    

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', default='./crnn/models/netCRNN63.pth')
    parser.add_argument('--image', default='/Users/zhangxin/gitlab/dmocr/test/pic/text_line_image')
    parser.add_argument('--suffix', default='jpg')
    args = parser.parse_args()
    return args

def read_boxfile(boxfile):
    line_txt = ''
    with open(boxfile, 'r') as fi:
        for line in fi:
            line = line.strip()
            arr = line.split(' ')
            line_txt += arr[0]
    return line_txt


def main(args):
    model,converter = crnnSource(args.model_path)
    if os.path.isfile(args.image):
        image = Image.open(args.image).convert('L')
        result = crnn_line_rec(model, converter, image)
        print result
    elif os.path.isdir(args.image):
        sum_char = 0
        sum_right = 0
        for filename in glob.glob(os.path.join(args.image, '*.'+args.suffix)):
            if filename[0] == '.':  continue
            boxfilename = filename[:-len(args.suffix)] + 'box'
            line_txt = read_boxfile(boxfilename)
            
            image = Image.open(filename).convert('L')
            result = crnn_line_rec(model, converter, image)

            # print type(line_txt), type(result)
            min_ed = editdistance.eval(line_txt.decode('utf-8'), result)
            print filename, boxfilename
            print '\t', line_txt
            print '\t', result
            print '\t', min_ed, len(line_txt.decode('utf-8')), len(line_txt.decode('utf-8')) - min_ed
            sum_char += len(line_txt.decode('utf-8'))
            sum_right += len(line_txt.decode('utf-8')) - min_ed
        print sum_char, sum_right, float(sum_right) / sum_char
        

if __name__ == '__main__':
    main(get_args())