import cv2 
 
# 视频文件输入初始化 
#filename = "meter1.mp4" 
#camera = cv2.VideoCapture(filename) 
 
# 视频文件输出参数设置 
'''out_fps = 12.0  # 输出文件的帧率 
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2') 
out1 = cv2.VideoWriter('v1.mp4', fourcc, out_fps, (500, 400)) 
out2 = cv2.VideoWriter('v2.mp4', fourcc, out_fps, (500, 400)) '''
 
# 初始化当前帧的前帧 
lastFrame = None 

no = 1
# 遍历视频的每一帧 
while no != 630:
 
    # 读取下一帧 
    #(ret, frame) = camera.read() 
    frame = cv2.imread('../430/' + str(no) + '.jpg')
    no += 1
 
    # 如果不能抓取到一帧，说明我们到了视频的结尾 
    #if not ret: 
    #    break 
    # 调整该帧的大小 
    frame = cv2.resize(frame, (500, 400), interpolation=cv2.INTER_CUBIC) 
 
    # 如果第一帧是None，对其进行初始化 
    if lastFrame is None: 
        lastFrame = frame 
        continue 
 
    # 计算当前帧和前帧的不同 
    frameDelta = cv2.absdiff(lastFrame, frame) 
 
    # 当前帧设置为下一帧的前帧 
    lastFrame = frame.copy() 
 
    # 结果转为灰度图 
    thresh = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY) 
 
    # 图像二值化 
    thresh = cv2.threshold(thresh, 15, 255, cv2.THRESH_BINARY)[1] 
 
    ''' 
    #去除图像噪声,先腐蚀再膨胀(形态学开运算) 
    thresh=cv2.erode(thresh,None,iterations=1) 
    thresh = cv2.dilate(thresh, None, iterations=2) 
    ''' 
 
    # 阀值图像上的轮廓位置 
    binary, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
 
    # 遍历轮廓 
    for c in cnts: 
        # 忽略小轮廓，排除误差 
        if cv2.contourArea(c) < 100: 
            continue 
 
        # 计算轮廓的边界框，在当前帧中画出该框 
        (x, y, w, h) = cv2.boundingRect(c) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
 
    # 显示当前帧 
    cv2.imshow("frame", frame) 
    cv2.imshow("frameDelta", frameDelta) 
    cv2.imshow("thresh", thresh) 
 
    # 保存视频 
    #out1.write(frame) 
    #out2.write(frameDelta) 
 
    # 如果q键被按下，跳出循环 
    if cv2.waitKey(200) & 0xFF == ord('q'): 
        break 
 
# 清理资源并关闭打开的窗口 
out1.release() 
out2.release() 
#camera.release() 
cv2.destroyAllWindows()