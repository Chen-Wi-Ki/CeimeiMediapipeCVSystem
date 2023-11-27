import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from picamera.array import PiRGBArray
from picamera import PiCamera



cap = cv2.VideoCapture(0)  # 開啟攝像頭，0代表預設攝像頭
cap.set(cv2.CAP_PROP_FPS, 5)
cap.set(3,320)
cap.set(4,480)
#cap.set(5,5)


a = 0
b = 0
c = 0

# 初始化Tkinter視窗
root = tk.Tk()
root.title("手術縫合偵測")
root.geometry("320x480")
root.overrideredirect(True) #全螢幕設置,配合自啟動程式使用

# 建立用於顯示影像的Label
label = tk.Label(root, width=320, height=480)
label.pack()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.4)

def angle(a, b, c):
    vector1 = np.array(a) - np.array(b)
    vector2 = np.array(c) - np.array(b)
    angle = np.degrees(np.arctan2(np.cross(vector1, vector2), np.dot(vector1, vector2)))
    return angle


def angle_x_y_z(a, b, c):
    ax = [0, a[1], a[2]]
    bx = [0, b[1], b[2]]
    cx = [0, c[1], c[2]]

    ay = [a[0], 0, a[2]]
    by = [b[0], 0, b[2]]
    cy = [c[0], 0, c[2]]

    az = [a[0], a[1], 0]
    bz = [b[0], b[1], 0]
    cz = [c[0], c[1], 0]

    xangle = angle(ax, bx, cx)
    yangle = angle(ay, by, cy)
    zangle = angle(az, bz, cz)
    return xangle[0], yangle[1], zangle[2]

def TextInPicture(image ,text,position,font_color,font_scale = 1.7,gray=10,gray_y=0):

    # 文字顏色和字型設置
    font = cv2.FONT_HERSHEY_SIMPLEX

    thickness = 2  # 文字線寬
    # 在影像上繪製文字

    # 獲取文字的大小
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 添加半透明灰色背景
    alpha = 0.7  # 背景的透明度，可自行調整
    overlay = image.copy()
    x, y = position
    width, height = text_size[0], text_size[1] + gray
    cv2.rectangle(overlay, (x, gray_y), (x + width, 0 + height), (60, 60, 60), -1)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.putText(image, text, position, font, font_scale, font_color, thickness)





the_massage = ""
the_massage3 = "Times: "


hand_position =0
sum_x = 0
count = 0
times = 0

def update_image():
    global a, b, c
    global the_massage
    global the_massage3
    global hand_position
    global sum_x ,count,times

    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)
    frame = cv2.resize(frame,(320,480))
    if not ret:
        print("無法讀取影像")
        return
    # 將影像轉換成RGB格式，因為Mediapipe使用的是RGB格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.flip(frame_rgb,-1)


    # 使用Mediapipe來偵測手
    results = hands.process(frame_rgb)



    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = []


            for id, lm in enumerate(hand_landmarks.landmark):
                landmark_list.append([lm.x, lm.y, lm.z])
                """
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame_rgb, (x, y), 5, (255, 0, 0), -1)
                """
            # 檢查是否取得了0、1和5這三個Landmark，如果有就進行計算
            if len(landmark_list) >= 6:

                positionx0, postitiony0, positionz0 = landmark_list[0]
                positionx9,postitiony9,positionz9= landmark_list[9]
                positionx = (positionx9+positionx0)/2.0
                sum_x = sum_x + positionx
                count = count+1

                if count == 18:
                    """
                    average_x = sum_x / count
                    count = 0
                    sum_x = 0
                    hand_position2 = hand_position - average_x * 1000
                    print( abs(hand_position2))
                    hand_position = positionx * 1000
                    # print(abs(hand_position2))

                    if hand_position2 != positionx:
                        if abs(hand_position2) >= 110:
                            TextInPicture(frame_rgb, "Warning: Parallel movement", (0, 50), (255, 0, 0), font_scale=1)
                            """




                landmark_2 = landmark_list[2]
                landmark_3 = landmark_list[3]
                landmark_5 = landmark_list[5]

                x, y, z = angle_x_y_z(landmark_3, landmark_2, landmark_5)
                the_massage2 = "Angle: " + str(int(x))
                TextInPicture(frame_rgb, the_massage2, (0, 50), (50, 255, 50),font_scale = 1,gray=25,gray_y=50)


                # print(x,y,z)
                if x > 1:
                    a = 1
                    # print("手心向上")
                else:
                    b = 1
                    # print("手心向下")

                if a == 0 and b == 1:
                    c = 1


                if a == 1 and b == 0:
                    c = 2


                if a == 1 and c == 1:
                    a = 0
                    b = 0
                    c = 0
                    #the_massage = "Flip up"
                    times = times+1
                    the_massage3 = "Times: "+str(times)





                if b == 1 and c == 2:
                    a = 0
                    b = 0
                    c = 0

                    #the_massage = "Flip down"
















    else:
        the_massage2 = "Undetected"
        TextInPicture(frame_rgb, the_massage2, (0, 50), (50, 255, 50), font_scale=1,gray=25,gray_y=50)


    """
    TextInPicture(frame_rgb,the_massage, (0, 25), (50, 255, 50),font_scale = 1)
    TextInPicture(frame_rgb, the_massage3, (0, 65), (50, 255, 50), font_scale=1,gray=50,gray_y=40)
    """
    TextInPicture(frame_rgb, the_massage3, (0, 25), (50, 255, 50),font_scale = 1)


    # 更新Tkinter視窗中的影像
    image = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image=image)
    label.config(image=photo)
    label.image = photo
    # 循環呼叫更新
    root.after(10, update_image)


# 開始更新影像和訊息
update_image()

# 啟動Tkinter主迴圈
root.mainloop()

# 釋放攝像頭資源
cap.release()
cap0.release()
cv2.destroyAllWindows()









"""
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
a=0
b=0
c=0
def angle(a, b, c):
    vector1 = np.array(a) - np.array(b)
    vector2 = np.array(c) - np.array(b)
    angle = np.degrees(np.arctan2(np.cross(vector1, vector2), np.dot(vector1, vector2)))
    return angle


def angle_x_y_z(a, b, c):
    ax = [0, a[1], a[2]]
    bx = [0, b[1], b[2]]
    cx = [0, c[1], c[2]]

    ay = [a[0], 0, a[2]]
    by = [b[0], 0, b[2]]
    cy = [c[0], 0, c[2]]

    az = [a[0], a[1], 0]
    bz = [b[0], b[1], 0]
    cz = [c[0], c[1], 0]

    xangle = angle(ax, bx, cx)
    yangle = angle(ay, by, cy)
    zangle = angle(az, bz, cz)
    return xangle[0], yangle[1], zangle[2]


while True:
    # Get image frame
    success, img = cap.read()
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # Find the hand and its landmarks
    #hands, img = detector.findHands(img)  # with draw
    img = cv2.flip(img, 1)
    hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        size = np.array(lmList1)
        #print(size)
        #print(size[0])
        x, y, z = angle_x_y_z(size[1],size[0],size[5])
        print(x)
        
        
        
        

        
        if z > 1:
            a=1
            #print("手心向上")
        else:
            b=1
            #print("手心向下")

        if a==0 and b==1:
            c=1
        if a == 1 and b == 0:
            c=2

        if a==1 and c==1:
            print("向上翻轉")
            a = 0
            b = 0
            c= 0
        
        if  b == 1 and c==2:
            print("向下翻轉")
            a = 0
            b = 0
            c= 0
        





        #bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        #centerPoint1 = hand1['center']  # center of the hand cx,cy
        #handType1 = hand1["type"]  # Handtype Left or Right
        #fingers1 = detector.fingersUp(hand1)
        # Display
        cv2.imshow("Image", img)
        cv2.waitKey(100)







cap.release()
cv2.destroyAllWindows()
"""
