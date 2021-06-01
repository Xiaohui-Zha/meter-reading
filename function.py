from numpy.lib.histograms import histogram
import tensorflow as tf
import os
import random
import numpy as np
import cv2

#------------------#
# single_track
# return a pic with a path
#------------------#

def trackSingle(type):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    # Read video
    video = cv2.VideoCapture("resizex1.mp4")

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    i = 0
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)
        #print(bbox)
        #global img
        #i = 0
        cv2.putText(frame, "frame : " + str(int(i)), (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        #cut_img = frame[int(bbox[0]):int(bbox[0]+bbox[2]), int(bbox[1]):int(bbox[1] + bbox[3])]
        #cv2.imwrite('cut_img'+ str(i) + '.jpg', cut_img)
        i += 1

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            if i % 5 == 0:
                crop = frame[int(bbox[1]):int(bbox[1]+bbox[3]),
                             int(bbox[0]):int(bbox[0]+bbox[2])]
                #cv2.imwrite('./pic_tmp/'+'crop'+str(i)+'.jpg', crop)
                #PicPath = './pic_tmp/' + 'crop' + str(i) + '.jpg'
                #cv2.imwrite(PicPath, crop)
                if type == 'light':
                    result = detection(crop)
                elif type == 'nums':
                    result = pred_nums(crop)

        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

#======================基于区域生长法，检测指示灯=========================
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y
 
def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
 
def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects
 
def regionGrow(img,seeds,thresh,p = 1):
    height, weight = img.shape
    seedMark = np.ones(img.shape,dtype=np.uint8)*255
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 0
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if grayDiff < thresh and seedMark[tmpX,tmpY] == 255:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark
 
def on_mouse(event, x,y, flags , params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + 'Point' + '('+str(x) + ', ' + str(y)+')')
        clicks.append((y, x))

def detection(img):
    lights = '无'
    img = cv2.resize(img,(200,int(200/img.shape[1]*img.shape[0])))
    img = cv2.GaussianBlur(img,(5,5),0,0)
    tmp = img.copy()

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    '''
    clicks=[]
    cv2.imshow('img',img)
    cv2.setMouseCallback('img', on_mouse )
    cv2.waitKey()
    '''

    # seeds = [Point(10,10),Point(82,150),Point(20,300)]
    # seeds = [Point(i[0],i[1]) for i in clicks]
    seeds = [Point(5,5)]

    binaryImg = regionGrow(img,seeds,10)
    # binaryImg = cv2.bitwise_and(binaryImg,img)
    # binaryImg = cv2.cvtColor(binaryImg , cv2.COLOR_)
    _,contours,hierarchy = cv2.findContours(binaryImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours: 
        (circle_x,circle_y),radius = cv2.minEnclosingCircle(contour)
        circle_center = (int(circle_x),int(circle_y))
        radius = int(radius)
        if radius <= 4 or radius > 10 :
            continue
        # print(radius)
        # print(circle_center)
        tmp1 = cv2.cvtColor(tmp ,cv2.COLOR_BGR2HSV)
        print('HSV颜色空间为{}'.format(tmp1[int(circle_y)][int(circle_x)]))
        cv2.circle(tmp ,circle_center ,radius ,(0,0,255) ,2)
        
        
        # cv2.imshow('regiongrow',binaryImg)
        # 显示指示灯的检测位置
        '''
        cv2.imshow('tmp',tmp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        hsv = tmp1[int(circle_y)][int(circle_x)][1]
        if 0 <= hsv <= 10 or 156 <= hsv <= 180:
            lights = '红'
        elif 11 <= hsv <= 25:
            lights = '橙'
        elif 26 <= hsv <= 34:
            lights = '黄'
        elif 35 <= hsv <= 77:
            lights = '绿'
        elif 78 <= hsv <= 99:
            lights = '青'
        elif 100 <= hsv <= 124:
            lights = '蓝'
        else:
            lights = '紫'
    return lights



#======================基于直方图切割，识别数字=========================
tar_temp=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.','A','C','E','F','H','L','P']

#输入二值图像，输出二值图像
#根据黑白占比判断是黑底白字/白底黑字
def bit_not(binary):
    height,width = binary.shape
    m = [height-1]*width
    M = [0]*width
    black,white = 0,0
    for j in range(width):
        for i in range(height):
            if binary[i][j] == 255:
                white += 1
                m[j] = min(m[j] ,i)
                M[j] = max(M[j] ,i)
            if binary[i][j] == 0:
                black += 1
    if black < white:
        binary = cv2.bitwise_not(binary)
    return binary

#输入RGB图，输出二值图像、竖直、水平投影
#画出水平、竖直直方图投影
def draw_hist(img):
    img = cv2.resize(img,(200,int(200/img.shape[1]*img.shape[0])))
    img = cv2.GaussianBlur(img,(5,5),0,0)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   

    ret,thresh=cv2.threshold(gray,130,255,cv2.THRESH_BINARY)  
    ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
    thresh = bit_not(thresh)


    thresh1=thresh.copy()
    thresh2=thresh.copy()
    h,w=thresh.shape

    row = [0]*w
    #记录每一列的波峰
    for j in range(w): #遍历一列 
        for i in range(h):  #遍历一行
            if  thresh1[i,j]==0:  #如果改点为黑点
                row[j]+=1  		#该列的计数器加一计数
                thresh1[i,j]=255  #记录完后将其变为白色 
            
    for j  in range(w):  #遍历每一列
        for i in range(row[j]):  #从该列应该变黑的最顶部的点开始向最底部涂黑
            thresh1[i,j]=0   #涂黑

    # plt.imshow(thresh1,cmap=plt.gray())
    # plt.show()

    col = [0]*h 
    for j in range(h):  
        for i in range(w):  
            if  thresh2[j,i]==0: 
                col[j]+=1 
                thresh2[j,i]=255
            
    for j  in range(h):  
        for i in range(w-col[j],w):   
            thresh2[j,i]=0    

    cv2.imshow('img',img)
    cv2.imshow('threshold',thresh)
    cv2.imshow('row',thresh1)  
    cv2.imshow('col',thresh2)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

#根据阈值、直方图，找出波峰
def find_waves(threshold,histogram):
    up_point = -1
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i,x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks

#输入水平波峰,截取水平方向的图片
#返回值为一对对角点（左上、右下）
def cut(img ):
    img = cv2.resize(img,(200,int(200/img.shape[1]*img.shape[0])))
    img = cv2.GaussianBlur(img,(5,5),0,0)
    gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    # _,binary = cv2.threshold(gray ,130 ,255 ,cv2.THRESH_BINARY)
    _,binary = cv2.threshold(gray ,0 ,255 ,cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    binary = bit_not(binary)
    height,width = binary.shape

    # binary = binary[1:height-1]
    y_histogram = np.sum(binary ,axis=0 )
    y_min = np.min(y_histogram)
    y_average = np.sum(y_histogram)/y_histogram.shape[0]
    y_threshold = (y_min + y_average)/5
    wave_peaks = find_waves(y_threshold ,y_histogram)
    points=[]
    for wave in wave_peaks:
        x1,x2 = wave

        im = binary[:,x1:x2]
        x_histogram  = np.sum(im, axis=1)
        x_min = np.min(x_histogram)
        x_average = np.sum(x_histogram)/x_histogram.shape[0]
        x_threshold = (x_min + x_average)/2
        # print(x_histogram)
        for ind in range(1,len(x_histogram)):
            if x_histogram[ind] - x_histogram[ind-1] > x_threshold/2:
                m = ind
                break
        for ind in range(len(x_histogram)-2,-1,-1):
            if x_histogram[ind] - x_histogram[ind+1] > x_threshold/2:
                M = ind
                break
        y1,y2 = m,M
        if m > M:
            continue
        im = im[y1:y2,:]
        point = [(x1,y1),(x2,y2)]

        '''
        cv2.rectangle(img ,point[0] ,point[1] ,(0,255,0) ,2)
        cv2.imshow(str(x1),im)
        cv2.waitKey()
        '''

        if x1 < 3 or x2 > width-3:
            continue
        points.append(point)

    '''
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    return points

def dilate_erode(binary):
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 4))
    kernel_dilate2 = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 1))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 1))
    binary = cv2.dilate(binary ,kernel_dilate ,iterations = 1)
    return binary

#轻微仿射变换，向左倾斜
def Affine(img):
    height,width = img.shape[:2]
    p2 = np.float32([ [width-1,height-1], [width-6,0], [0,height-1] ])
    p1 = np.float32([ [width-1,height-1], [width-1,0], [0,height-1] ])
    M = cv2.getAffineTransform( p1, p2)
    dst = cv2.warpAffine( img, M, (width,height))
    return dst


def pred_nums(img):
    nums = '无'
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('../datasets/models/my_digit_model.meta')
        new_saver.restore(sess ,'../datasets/models/my_digit_model')
        yy_hyp = tf.get_collection('yconv')[0]
        graph = tf.get_default_graph() 
        X = graph.get_operation_by_name('X').outputs[0]#为了将 x placeholder加载出来
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0] # 将keep_prob placeholder加载出来
        
        # mm用来保存数字以及数字坐标
        mm={}

        img = Affine(img)
        img = cv2.resize(img,(200,int(200/img.shape[1]*img.shape[0])))
        img = cv2.GaussianBlur(img,(5,5),0,0)
        gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
        # _,binary = cv2.threshold(gray ,130 ,255 ,cv2.THRESH_BINARY)
        _,binary = cv2.threshold(gray ,0 ,255 ,cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
        binary = bit_not(binary)
        
        points = cut(img)
        for point in points:
            x1,y1 = point[0]
            x2,y2 = point[1]
            im = binary[y1-2:y2+2,x1-2:x2+2]
            if y1-2<0 or y2+2>binary.shape[0]-1 or x1-2<0 or x2+2>binary.shape[1]:
                im = binary[y1: y2, x1:x2]
            im = dilate_erode(im)
            roi = cv2.resize(im ,(28,28) ,interpolation=cv2.INTER_AREA)
            roi = np.array([roi.reshape(28*28)/255])
            pred = sess.run(yy_hyp ,feed_dict = {X:roi,keep_prob:1.0})
            ind = np.argmax(pred)
            if ind==10 and (y2-y1)/(x2-x1)>1.5:ind=1
            if ind==1 and (y2-y1)/(x2-x1)<1.5:ind=10
            mm[x1] = tar_temp[ind]
        num_tup = sorted(mm.items() ,key=lambda x:x[0])
        nums = ''.join([i[1] for i in num_tup])
        # print('实际结果为：',curr.split('_')[1][:-4])
        # print('预测结果为：',nums)
        '''
        cv2.imshow('img' ,img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        '''
        mm.clear()
    return nums