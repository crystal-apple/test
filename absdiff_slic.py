from skimage.segmentation import slic, mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import cv2
from timeit import default_timer as timer


# np.set_printoptions(threshold=np.inf)

def find_boundaries(segments, coordinate):  # coordinate:[x,y,w,h]
    img_h, img_w = segments.shape
    y, x, w, h = coordinate
    # if direction == "left":
    min_left = y
    for i in range(x, x + h):
        for j in range(y, -1, -1):
            if segments[i][j] == segments[i][y]:
                if j < min_left:
                    min_left = j
            else:
                break
    #    return min_left
    # if direction == "right":
    max_right = y + w - 1
    for i in range(x, x + h):
        for j in range(y + w, img_w):
            if segments[i][j] == segments[i][y + w]:
                if j > max_right:
                    max_right = j
            else:
                break
    #    return max_right
    # if direction == "up":
    min_up = x
    for i in range(y, y + w):
        for j in range(x, -1, -1):
            if segments[j][i] == segments[x][i]:
                if j < min_up:
                    min_up = j
            else:
                break
    #    return min_up
    # if direction == "down":
    max_down = x + h - 1
    for i in range(y, y + w):
        for j in range(x + h - 1, img_h):
            if segments[j][i] == segments[x + h - 1][i]:
                if j > max_down:
                    max_down = j
            else:
                break

    return [min_left, min_up, int(max_right - min_left), int(max_down - min_up)]


def aggregateboxes(rects):
    rectNumber = 0
    while rectNumber < len(rects):
        rectNumber += 1
        flag = True
        while flag:
            flag = False
            xmin = rects[rectNumber - 1][0]
            ymin = rects[rectNumber - 1][1]
            xmax = xmin + rects[rectNumber - 1][2] - 1
            ymax = ymin + rects[rectNumber - 1][3] - 1
            i = rectNumber - 2
            while i > 0:
                #    print('iiiiii',i)
                i -= 1
                ixmin = rects[i][0]
                iymin = rects[i][1]
                ixmax = ixmin + rects[i][2] - 1
                iymax = iymin + rects[i][3] - 1
                if xmax < ixmin - 1 - 20 or xmin > ixmax + 1 + 20 or ymax < iymin - 1 - 20 or ymin > iymax + 1 + 20:
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
                # rects.pop(-1)
                rectNumber -= 1;
            # break
    return rects


def absdiff_demo(image_1, image_2, sThre):
    gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_image_1 = cv2.GaussianBlur(gray_image_1, (3, 3), 0)
    gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    gray_image_2 = cv2.GaussianBlur(gray_image_2, (3, 3), 0)
    d_frame = cv2.absdiff(gray_image_1, gray_image_2)
    ret, d_frame = cv2.threshold(d_frame, sThre, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(d_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def IROI_dection(background, current_frame, sThre=30):
    boxes = []
    contours = absdiff_demo(background, current_frame, sThre)
    for index in range(len(contours)):
        rect = cv2.boundingRect(contours[index])
        if (rect[2] < 10 or rect[3] < 10) and (rect[2] != 0 or rect[3] != 0): #CaVignal
        #if (rect[2] < 50 or rect[3] < 50) and (rect[2] != 0 or rect[3] != 0):
            continue
        else:
            boxes.append(rect)
    boxes = aggregateboxes(boxes)

    if len(boxes) > 0:
        S = []
        for i in range(len(boxes)):
            s = boxes[i][2] * boxes[i][3]
            S.append(s)
        # num = S.index(min(S))
        num = S.index(max(S))
        coordinate = boxes[num]
        segments = slic(current_frame, n_segments=1500, compactness=10)
        boxes = find_boundaries(segments, coordinate)
        # return True,(boxes[0],boxes[1]),(int(boxes[0]+boxes[2]),int(boxes[1]+boxes[3]))
        return True, (boxes[0], boxes[1], boxes[2], boxes[3])
    else:
        return False, (0, 0, 0, 0)


if __name__ == "__main__":
    capture = cv2.VideoCapture("../data/CaVignal.avi")
    sThre = 30  # sThre表示像素阈值
    i = 0
    rects = []
    frame = cv2.imread('../data/back-CaVignal.jpg')
    # print('type', type(frame),frame.shape)
    ret = True
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    a = 1
    count = 0
    detect_num = 0
    sum_fps = 0
    # print('type-sum_fps', type(sum_fps))
    avg_fps = 0
    while True:
        ret_2, frame_2 = capture.read()
        count += 1
        #print(count)
        if ret_2 == True:
            if count > 0:
                #count += 1
                # if i == 0:
                # #    cv2.waitKey(660)
                #     i = i + 1
                # ret_2, frame_2 = capture.read()
                # print('type', type(frame_2),frame_2.shape, ret_2)
                contours = absdiff_demo(frame, frame_2, sThre)
                num = len(contours)
                for index in range(num):
                    rect = cv2.boundingRect(contours[index])
                    #if (rect[2] < 50 or rect[3] < 50) and (rect[2] != 0 or rect[3] != 0):
                    if (rect[2] < 10 or rect[3] < 10) and (rect[2] != 0 or rect[3] != 0): #CaVignal
                        continue
                    else:
                        rects.append(rect)
    
                #        cv2.rectangle(frame_2,(rect[0],rect[1]),(int(rect[0]+rect[2]),int(rect[1]+rect[3])),(0,0,255),2)
                # cv2.imshow('frame',frame_2)
                # cv2.imshow('contours',contours)
                # print(rects)
                rects = aggregateboxes(rects)
    
                # print('rects',rects)
                segments = slic(frame_2, n_segments=1500, compactness=10)
                segments = np.array(segments)
                if len(rects) > 0:
                    # print('len(rects)',len(rects))
                    S = []
                    for i in range(len(rects)):
                        s = rects[i][2] * rects[i][3]
                        S.append(s)
                    num = S.index(max(S))
                    coordinate = rects[num]
                    rects = find_boundaries(segments, coordinate)
                else:
                    #print("llllllllllllll")
                    # print(rects)
                    continue
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    a = a + 1
    
                    sum_fps = int(curr_fps) + sum_fps
    
                    curr_fps = 0
                    avg_fps = sum_fps / a
                print(count)
                # if count > 10:
                    # print(avg_fps)
                # print(rects)
                # cv2.rectangle(frame_2,(rect[0],rect[1]),(int(rect[0]+rect[2]),int(rect[1]+rect[3])),(0,0,255),2)
                cv2.rectangle(frame_2, (rects[0], rects[1]), (int(rects[0] + rects[2]), int(rects[1] + rects[3])),
                            (0, 0, 255), 2)
                cv2.putText(frame_2, "absdiff-slic", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
                cv2.putText(frame_2, text=fps, org=(3, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.75, color=(255, 0, 0), thickness=2)
                # cv2.rectangle(frame_2,(coordinate[0],coordinate[1]),(int(coordinate[0]+coordinate[2]),int(coordinate[1]+coordinate[3])),(0,0,255),2)
                # cv2.imshow('frame',frame_2)
                # for j in range(len(rects)):   
                # p1 = (int(rects[j][0]), int(rects[j][1]))
                # p2 = (int(rects[j][0] + rects[j][2]), int(rects[j][1] + rects[j][3]))
                # #print(p1,p2)
                # cv2.rectangle(frame_2, p1, p2, (0,0,255), 2)
                cv2.imshow('frame',frame_2)
                #cv2.imwrite('./result/absdif-slic-CaVignal/'+str(count)+'.jpg', frame_2)
                del rects[:]
    
                # frame = frame_2
            else:
                #count += 1
                continue
                # break
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            print('avg_fps', avg_fps)
            break
    capture.release()
'''
# segments = [[0,0,1,1,1],[0,0,1,1,1],[2,2,1,1,1],[2,2,3,3,3],[4,4,4,4,4],[5,5,5,5,5]]
# segments = np.array(segments)
# coordinate = [2,1,3,4]
# direction = "down"
# print(find_boundaries(segments,coordinate,direction))

# segments = slic(img, n_segments=60, compactness=10)
# out=mark_boundaries(img,segments)
# print(type(out))
# plt.subplot(121)
# plt.title("n_segments=60")
# plt.imshow(out)
#
# segments2 = slic(img, n_segments=300, compactness=10)
# out2=mark_boundaries(img,segments2)
# plt.subplot(122)
# plt.title("n_segments=300")
# plt.imshow(out2)
#
# plt.show()
'''
