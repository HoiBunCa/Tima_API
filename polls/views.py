from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from inference import detect_corner
from imutils import contours
from skimage import measure
from tqdm import tqdm
from PIL import Image
from PIL import ImageChops
from PIL import ImageEnhance
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler

import time
import numpy as np
import cv2
import imutils
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy


model = torch.load("model_ela.pt", map_location='cpu')
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = model.to(device)
class_names = ['fake', 'real']
acc = torch.nn.Softmax(dim=1)


def inference_img(filename):
    basename, extension = os.path.splitext(filename)
    resaved = 'resaved.jpg'
    ela = 'ela.png'
    im = Image.open(filename)
    im.save(resaved, 'JPEG', quality=90)
    resaved_im = Image.open(resaved)

    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    ela_im.save(ela)
    img_test = Image.open(ela)
    img_transforms = data_transforms(img_test)
    img_unsquueeze = img_transforms.unsqueeze(0).to(device)
    model.eval().to(device)
    output = model(img_unsquueeze)

    _, preds = torch.max(output, 1)
    return class_names[int(preds)], max(max(acc(output))).item()


def check_img_photo(path_img):
    img = cv2.imread(path_img)

    img = cv2.resize(img, (640, 480))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0
    # print("AAAAAAAAAA", np.mean(output_img))
    if np.mean(output_img) > 0:
        check = "Ảnh màu"
    else:
        check = "Ảnh photo"
    return check


def check_img_photo_1(path_img):
    img = cv2.imread(path_img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_sat = hsv[:, :, 1]

    # find frequency of pixels in range 0-255
    histr = cv2.calcHist([hsv_sat], [0], None, [256], [0, 256])

    thresh = sum(histr[0:50]) / sum(histr[0:])[0]
    if thresh < 0.96:
        result = "Ảnh màu"
    else:
        result = "Ảnh photo"
    print("AAAAAAAAAA", result)
    print("BBBBBBBBBB", thresh[0])
    return result, thresh


def check_overexposed(path_img):
    image = cv2.imread(path_img)
    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if numPixels > 300:
            mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) != 0:
        return "Có"
    else:
        return "Không"


@csrf_exempt
# Create your views here.
def predict(request):
    if request.method == 'POST':
        t1 = time.time()
        file = request.FILES.get('file')
        PIL_img = Image.open(file)
        arr_img = np.array(PIL_img)
        arr_img = cv2.cvtColor(arr_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("img_post.jpg", arr_img)
        result = detect_corner(['img_post.jpg'])
        print("result", result)

        cv2.imwrite("result.jpg", result[0])
        check1, thresh = check_img_photo_1("result.jpg")
        check2 = check_overexposed("result.jpg")
        check3, score = inference_img("img_post.jpg")
        time_process = time.time() - t1
        kq_api = {
            'Kiểm tra mất góc': result[1],
            'Kiểm tra ảnh photo': check1,
            'Kiểm tra ảnh lóa sáng': check2,
            'Kiểm tra ảnh thật - giả': check3 + str(" - {:.2f}".format(score)),
            'time': time_process,
        }

        # if result[1] == "Không":
        #     cv2.imwrite("result.jpg", result[0])
        #     check1, thresh = check_img_photo("result.jpg")
        #     check2 = check_overexposed("result.jpg")
        #     check3, score = inference_img("img_post.jpg")
        #     time_process = time.time() - t1
        #     kq_api = {
        #             'Kiểm tra mất góc': result[1],
        #             'Kiểm tra ảnh photo': check1,
        #             'Kiểm tra ảnh lóa sáng': check2,
        #             'Kiểm tra ảnh thật - giả': check3 + str(" - {:.2f}".format(score)),
        #             'time': time_process,
        #         }
        # if result[1] == "Có":
        #     cv2.imwrite("result.jpg", result[0])
        #     kq_api = {
        #             'Kiểm tra mất góc': "Có",
        #         }
        #
        # if result[1] == "Không phát hiện được góc":
        #     cv2.imwrite("result.jpg", result[0])
        #     check3, score = inference_img("img_post.jpg")
        #     kq_api = {
        #             'Kiểm tra mất góc': "Không phát hiện được góc",
        #             'Kiểm tra ảnh thật - giả': check3 + str(" - {:.2f}".format(score)),
        #         }
    else:
        kq_api = {}

    return JsonResponse(kq_api)