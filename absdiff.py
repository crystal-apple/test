import cv2
#帧差结果进行聚合
def aggregateboxes(rects):
    rectNumber = 0
    while rectNumber < len(rects):
        rectNumber += 1
        flag = True
        while flag:
            flag = False
            xmin = rects[rectNumber -1][0]
            ymin = rects[rectNumber -1][1]
            xmax = xmin + rects[rectNumber-1][2]-1
            ymax = ymin + rects[rectNumber-1][3]-1
            i = rectNumber - 2
            while i>0: 
               
                i -=1
                ixmin = rects[i][0]
                iymin = rects[i][1]
                ixmax = ixmin + rects[i][2]-1
                iymax = iymin + rects[i][3]-1
                if xmax < ixmin-1-10 or xmin > ixmax+1 + 10 or ymax < iymin - 1 - 10 or ymin > iymax + 1 + 10:
                    continue
                flag = True
                xmin = min(xmin, ixmin)
                xmax = max(xmax, ixmax)
                ymin = min(ymin, iymin)
                ymax = max(ymax, iymax)
                rects[i] = rects[rectNumber - 2]
                
                rects[rectNumber - 2] = (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)
                rects[rectNumber - 1] = rects[len(rects) - 1]
                del rects[-1]
                #rects.pop(-1) 
                rectNumber -= 1;
                #break
    return rects


def absdiff_demo(image_1, image_2, sThre):
    gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)  #灰度化
    gray_image_1 = cv2.GaussianBlur(gray_image_1, (3, 3), 0)  #高斯滤波
    gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    gray_image_2 = cv2.GaussianBlur(gray_image_2, (3, 3), 0)
    d_frame = cv2.absdiff(gray_image_1, gray_image_2)
    ret, d_frame = cv2.threshold(d_frame, sThre, 255, cv2.THRESH_BINARY)
    contours , hierarchy = cv2.findContours ( d_frame , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    return contours

def detect(background, current_frame, sThre):
    boxes = []

    contours = absdiff_demo(background, current_frame,sThre)
    for index in range(len(contours)):
        rect = cv2.boundingRect(contours[index])
        if (rect[2]<50 or rect[3]<50) and (rect[2]!=0 or rect[3]!=0):
            continue
        else:
            boxes.append(rect)
    boxes = aggregateboxes(boxes)
   
    return True,boxes



if __name__ == "__main__":    
    capture = cv2.VideoCapture("/home/zhangmeiling/dataset/multiren.avi")
    #capture = cv2.VideoCapture("/home/zhangmeiling/dataset/test4.mp4")
    sThre = 10 #sThre表示像素阈值
  
    #rects=[]
    #ret,frame = capture.read()
    #frame = cv2.imread('/home/zhangmeiling/dataset/background/multiren-background.jpg')
    ret, frame = capture.read()
    tempframe = frame

    while True:
        ret, frame_2 = capture.read()
        if ret != True:
            break
        ret, rects = detect(tempframe,frame_2,sThre)
        tempframe = frame_2
        if len(rects) > 0:
            for i in range(len(rects)):
                cv2.rectangle(frame_2, (rects[i][0], rects[i][1]), (int(rects[i][0] + rects[i][2]), int(rects[i][1] + rects[i][3])),
                        (0, 0, 255), 2)
            cv2.imshow('frame',frame_2)
        
        if cv2.waitKey(25)&0xFF == ord('q'):
            break
    capture.release()

