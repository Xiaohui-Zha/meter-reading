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

def addLightContrast(img, a, b):
    row, col, channel = img.shape
    for i in range(row):
        for j in range(col):
            for k in range(channel):
                color = img[i][j][k] * a + b
                if color >= 255:
                    img[i][j][k] = 255
                else:
                    img[i][j][k] = color
    
    return img

def random_gen_picture(ttf_path, idx, files, chars, s):

    img_path = random.choice(files)
    canvas = Image.open(img_path)
    canvas = canvas.convert("RGB")
    x, y = canvas.size
    
    def write_number():
        nonlocal canvas
        # 随机字体大小
        text_size = random.randint(10, 150)
        # 随机选择字体
        font_path = random.choice(ttf_path)
        font = ImageFont.truetype(font_path, text_size)
        # 随机选择字符串长度
        flag = random.randint(0, 1)
        size = 1
        if flag: size = random.randint(1, 6)

        # 随机生产该长度的字符串
        def add_char(a, b, input_string, text_width, text_height, min_offset_y):
            input_idx = random.randint(a, b)
            input_string += chars[input_idx]
            offset_x, offset_y = font.getoffset(chars[input_idx])
            min_offset_y = min(min_offset_y, offset_y)
            text_width += font.getsize(chars[input_idx])[0]
            text_height = max(text_height, font.getsize(chars[input_idx])[1])
            return input_string, text_width, text_height, min_offset_y

        input_string = ""
        text_width, text_height = 0, 0
        min_offset_y = float('inf')

        flag = random.randint(0, 3) # 随机前缀
        if not flag: input_string, text_width, text_height, min_offset_y = add_char(11, len(chars) - 1, input_string, text_width, text_height, min_offset_y)
        
        for _ in range(size):
            input_string, text_width, text_height, min_offset_y = add_char(0, 10, input_string, text_width, text_height, min_offset_y)

        # flag = random.randint(0, 3) # 随机后缀
        # if not flag: input_string, text_width, text_height = add_char(11, len(chars) - 1, input_string, text_width, text_height)
        
        # 画图
        if text_height > y or text_width > x: return 
        draw = ImageDraw.Draw(canvas)
        start = (random.randint(0, x - text_width), random.randint(0, y - text_height)) # 随机选择位置
        color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))] # 随机颜色
        draw.text(start, input_string, font=font, fill=random.choice(color))
        
        # 保存标签信息
        #text_height -= min_offset_y
        #x_center, y_center = (start[0] + text_width / 2) / x, (start[1] + min_offset_y + text_height / 2) / y
        x_center, y_center = (start[0] + text_width / 2) / x, (start[1] + text_height / 2) / y
        width, height = text_width / x, text_height / y
        with open(f'/home/shana/detection/digit/labels/' + s + f'/{idx}.txt', 'a') as f:
            f.write(f"{0} {x_center} {y_center} {width} {height}\n")

    # 随机增添字符串
    size = random.randint(1, 3)
    for _ in range(size):
        write_number()
    

    # 随机增加噪声
    canvas = np.array(canvas)
    flag = random.randint(0, 8)

    if flag == 4:
        canvas = tfactor(canvas)

    if flag == 8:
        canvas = addNoise(canvas)
    
    # 随机增加亮点
    size = random.randint(1, 2)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)

    
    """    # 生成椭圆
    for _ in range(size):
        color = random.randint(0, 180)
        satruration = 0
        light = 255
        radius = random.randint(5, 45)

        x_center, y_center = random.randint(radius, x - radius), random.randint(radius, y - radius)
        canvas = cv2.ellipse(canvas, (x_center, y_center), (radius, radius), random.randint(0, 360),0,360, (color, satruration, light), -1)

        x_center, y_center = x_center / x, y_center / y
        width, height = radius / x, radius / y
        with open(f'/home/shana/detection/digit/labels/' + s + f'/{idx}.txt', 'a') as f:
            f.write(f"{1} {x_center} {y_center} {width} {height}\n")"""
    # 保存图像
    canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR)
    #canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'/home/shana/detection/digit/images/' + s + f'/{idx}.jpg', canvas)
    

if __name__ == "__main__":
    # 字体文件和背景文件所在路径
    ttf_path = r'/home/shana/detection/digit/font/'
    back_path = r'/home/shana/detection/coco/images/val2017/'
    ttfs = [ttf_path + f for f in os.listdir(ttf_path)]
    files = [back_path + f for f in os.listdir(back_path) if f[-3:] == 'jpg']
    chars = ['0','1','2','3','4','5','6','7','8','9','.','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','U','R','S','T','U','V','W','X','Y','Z']
    for i in range(10):
        random_gen_picture(ttfs, i, files, chars, "test")

    for i in range(1000):
        random_gen_picture(ttfs, i, files, chars, "val")

    for i in range(5000):
        random_gen_picture(ttfs, i, files, chars, "train")
