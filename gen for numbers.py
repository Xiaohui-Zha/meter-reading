from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
from math import *
import random
import os

def r(val):
    return int(np.random.random() * val)

# 添加仿射变换
def rot(img,angel,shape,max_angel):

    size_o = [shape[1],shape[0]]

    size = (shape[1]+ int(shape[0]*cos((float(max_angel )/180) * 3.14)),shape[0])


    interval = abs( int( sin((float(angel) /180) * 3.14)* shape[0]))

    pts1 = np.float32([[0,0] ,[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):

        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,size)

    return dst

#添加透视畸变
def rotRandrom(img, factor, size):

    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [ r(factor), shape[0] - r(factor)], [shape[1] - r(factor),  r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst


#增加饱和度光照的噪声
def tfactor(img):

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2)
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7)
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8)

    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img


#添加高斯模糊
def AddGauss(img, level):

    return cv2.blur(img, (level * 2 + 1, level * 2 + 1))


#添加高斯噪声
def AddNoiseSingleChannel(single):

    diff = 255-single.max()
    noise = np.random.normal(0,1+r(6),single.shape)
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise= diff*noise
    noise= noise.astype(np.uint8)
    dst = single + noise
    return dst

def addNoise(img,sdev = 0.5,avg=10):
    img[:,:,0] =  AddNoiseSingleChannel(img[:,:,0])
    img[:,:,1] =  AddNoiseSingleChannel(img[:,:,1])
    img[:,:,2] =  AddNoiseSingleChannel(img[:,:,2])
    return img
"""
def random_gen_picture(ttf_path, idx, files, chars, s):

    img_path = random.choice(files)
    canvas = Image.open(img_path)
    canvas = canvas.convert("RGB")
    x, y = canvas.size


    # 随机增添字符串
    start = [random.randint(0, x // 2), random.randint(0, y // 2)] # 随机选择位置
    start_copy = start.copy()
    text_size = random.randint(30, 200) # 随机字体大小
    size = random.randint(1, 10) # 随机长度
    bbox = []
    text_width, text_height = 0, 0
    for _ in range(size):
        # 随机选择字体
        font_path = random.choice(ttf_path)
        font = ImageFont.truetype(font_path, text_size)

        input_string = ""
        input_idx = random.randint(0, 10)
        input_string += chars[input_idx]
        offset_x, offset_y = font.getoffset(chars[input_idx])
        xmin, ymin, xmax, ymax = font.getmask(chars[input_idx]).getbbox()

        if ((start[1] + max(text_height, font.getsize(chars[input_idx])[1])) > y) or ((start[0] + text_width  + font.getsize(chars[input_idx])[0])> x): break
        text_width += font.getsize(chars[input_idx])[0]
        text_height = max(text_height, font.getsize(chars[input_idx])[1])
       
        # 画图
        draw = ImageDraw.Draw(canvas)
        
        color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))] # 随机颜色
        draw.text(start, input_string, font=font, fill=random.choice(color))
        
        # 保存标签信息
        # print(offset_x, offset_y)
        x_center, y_center = (start[0] - start_copy[0] + (font.getsize(chars[input_idx])[0]) / 2), text_height / 2
        width, height = font.getsize(chars[input_idx])[0], text_height
        bbox.append((input_idx, x_center, y_center, width, height))
        
        start[0] += font.getsize(chars[input_idx])[0]
    
    canvas = np.array(canvas)
    # print(start_copy, text_height, text_width)
    canvas = canvas[start_copy[1]:start_copy[1] + text_height, start_copy[0]:start_copy[0] + text_width, :]
    if len(canvas) == 0: return 
    with open(f'/home/shana/detection/digit/labels/' + s + f'/{idx}.txt', 'a') as f:
        for input_idx, x_center, y_center, width, height in bbox:
            f.write(f"{input_idx} {x_center / text_width} {y_center / text_height} {width / text_width} {height / text_height}\n")
    
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'/home/shana/detection/digit/images/' + s + f'/{idx}.jpg', canvas)"""

def random_gen_picture(ttf_path, idx, files, chars, s):
    img_path = random.choice(files)
    canvas = Image.open(img_path)
    canvas = canvas.convert("RGB")
    x, y = canvas.size


    # 随机增添字符
    font_path = random.choice(ttf_path) # 随机字体
    text_size = random.randint(40, 180) # 随机字体大小
    font = ImageFont.truetype(font_path, text_size)
    size = random.randint(2, 7)
    flag = random.randint(0, 3) # 增加单个字符图像出现的概率 
    if not flag:
        size = 1
        text_size = random.randint(40, 250)
    max_text_width, max_text_height = 0, 0
    for i in range(11):
        text_width, text_height = font.getsize(chars[i])
        max_text_height = max(max_text_height, text_height)
        max_text_width  = max(max_text_width, text_width)
    max_text_height *= 1.1 # 底部留空
    if x < max_text_width *size or y < max_text_height: return 

    start = [random.randint(0, x - max_text_width * size), random.randint(0, y - max_text_width)]
    canvas = canvas.crop((start[0], start[1], start[0] + max_text_width * size, start[1] + max_text_height))
    x, y = canvas.size
    w = 0
    for _ in range(size):
        if size == 1:
            input_idx = random.randint(0, 9)
        else:
            input_idx = random.randint(0, 10)
        
        offset_x, offset_y = font.getoffset(chars[input_idx])
        xmin, ymin, xmax, ymax = font.getmask(chars[input_idx]).getbbox()
        text_width, text_height = font.getsize(chars[input_idx])
        start = [w, 0]
        
        draw = ImageDraw.Draw(canvas)
        color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))] # 随机颜色
        draw.text(start, chars[input_idx], font=font, fill=random.choice(color))
        x_center, y_center = (w + text_width / 2) / x, (offset_y + (text_height - offset_y) / 2) / y
        width, height = text_width / x, (text_height - offset_y) / y
        # print(xmax - xmin, ymax - ymin)
        with open(f'/home/shana/detection/digit/labels/' + s + f'/{idx}.txt', 'a') as f:
            f.write(f"{input_idx} {x_center} {y_center} {width} {height}\n")
        
        w += text_width
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    canvas = np.array(canvas)
    flag = random.randint(0, 9)
    if not flag:
        canvas = tfactor(canvas)

    flag = random.randint(0, 9)
    if not flag:
        canvas = addNoise(canvas)
    
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'/home/shana/detection/digit/images/' + s + f'/{idx}.jpg', canvas)
    

if __name__ == "__main__":
    # 字体文件和背景文件所在路径
    ttf_path = r'/home/shana/detection/digit/font/'
    back_path = r'/home/shana/detection/coco/images/val2017/'
    ttfs = [ttf_path + f for f in os.listdir(ttf_path)]
    files = [back_path + f for f in os.listdir(back_path) if f[-3:] == 'jpg']
    chars = ['0','1','2','3','4','5','6','7','8','9','.']
    #for i in range(10):
    #    random_gen_picture(ttfs, i, files, chars, "test")

    for i in range(1000):
        random_gen_picture(ttfs, i, files, chars, "val")

    for i in range(5000):
        random_gen_picture(ttfs, i, files, chars, "train")

