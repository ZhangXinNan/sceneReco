#encoding=utf-8
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
            print '\t', result.encode('utf-8') # 此处直接打印result会报错，报错信息见最后
            print '\t', min_ed, len(line_txt.decode('utf-8')), len(line_txt.decode('utf-8')) - min_ed
            sum_char += len(line_txt.decode('utf-8'))
            sum_right += len(line_txt.decode('utf-8')) - min_ed
        print sum_char, sum_right, float(sum_right) / sum_char
        

if __name__ == '__main__':
    main(get_args())

'''
➜  sceneReco git:(zxdev) ✗ python crnnport.py > 20171227_netCRNN63.pth.log
Traceback (most recent call last):
  File "crnnport.py", line 172, in <module>
    main(get_args())
  File "crnnport.py", line 164, in main
    print '\t', str(result)
UnicodeEncodeError: 'ascii' codec can't encode characters in position 0-23: ordinal not in range(128)
'''