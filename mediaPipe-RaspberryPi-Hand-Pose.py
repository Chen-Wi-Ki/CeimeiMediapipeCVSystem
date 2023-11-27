import cv2
cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,480)
cap.set(5,8)

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils #mediapipe圖形繪製API
mp_hands = mp.solutions.hands #mediapipe手部API

from gpiozero import Button, RGBLED
led = RGBLED(red=26,green=19,blue=13) #RGD LED設定
button = Button(12) #按鍵設定

import threading
thread_Flag1 = True #用於判斷是否結束所有迴圈

import numpy as np
x = 0
y = 0
z = 0
xx= 0
yy= 0
zz= 0
cal_array=[]

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

    az = [a[0], a[1],0]
    bz = [b[0], b[1], 0]
    cz = [c[0], c[1], 0]

    xangle = angle(ax, bx, cx)
    yangle = angle(ay, by, cy)
    zangle = angle(az, bz, cz)

    return xangle[0],yangle[1], zangle[2]

def TextInPicture(image ,text,position,font_color,font_scale = 1.7,gray=10,gray_y=0):

    # 文字顏色和字型設置
    font = cv2.FONT_HERSHEY_SIMPLEX

    thickness = 1  # 文字線寬
    # 在影像上繪製文字

    # 獲取文字的大小
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 添加半透明灰色背景
    alpha = 0  # 背景的透明度，可自行調整 0.7
    overlay = image.copy()

    x, y = position
    width, height = text_size[0], text_size[1] + gray
    cv2.rectangle(overlay, (x, gray_y), (x + width, 0 + height), (60, 60, 60), -1)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.putText(image, text, position, font, font_scale, font_color, thickness)

def Cumulative_calculation(value):
    global cal_array
    total_difference = 0
    cal_array = cal_array + [value]
    if len(cal_array) == 7:
        total_difference = calculate_differences_and_sum(cal_array)
        cal_array = []
    return total_difference

def calculate_differences_and_sum(data):
    differences = [0] * (len(data) - 1)
    total_difference = 0

    for i in range(1, len(data)):
        differences[i - 1] = data[i] - data[i - 1]
    total_difference = sum(differences)
    return total_difference

#----------------#
#主呼叫由這裡開始#
#----------------#
def Run_Mediapipe():
    global thread_Flag1
    with mp_hands. Hands (
        static_image_mode = False,
        max_num_hands = 1,
        min_detection_confidence = 0.5 ) as hands:
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    #顯示辨識
                    #mp_drawing.draw_landmarks(
                    #    frame, hand_landmarks, #mp_hands.HAND_CONNECTIONS,
                    #    mp_drawing.DrawingSpec(color=(0,255,255), thickness=3,
                    #                                           circle_radius=10),
                    #    mp_drawing.DrawingSpec(color=(255,0,255), thickness=4,
                    #                                           circle_radius=1)
                    #)
                    #判斷角度
                    landmark_list = []

                    for id, lm in enumerate(hand_landmarks.landmark):
                        landmark_list.append([lm.x, lm.y, lm.z])

                    # 檢查是否取得了0、1和5這三個Landmark，如果有就進行計算
                    if len(landmark_list) >= 6:

                        positionx0, postitiony0, positionz0 = landmark_list[0]
                        positionx9,postitiony9,positionz9= landmark_list[9]
                        positionx = (positionx9+positionx0)/2.0

                        landmark_2 = landmark_list[2]
                        landmark_3 = landmark_list[3]
                        landmark_5 = landmark_list[5]

                        #global xx,yy,zz,message

                        x,y,z = angle_x_y_z(landmark_3, landmark_2, landmark_5)
                        xx = Cumulative_calculation(x)
                        yy = Cumulative_calculation(y)
                        zz = Cumulative_calculation(z)
                message1 = 'AngX='+str(int(x))+','+str(int(xx))
                message2 = 'AngY='+str(int(y))+','+str(int(yy))
                message3 = 'AngZ='+str(int(z))+','+str(int(zz))
            else:
                message1 = 'AngX dissonance'
                message2 = 'AngY dissonance'
                message3 = 'AngZ dissonance'
            cv2.putText(frame, message1, (1, 30), cv2.FONT_HERSHEY_PLAIN,1,
                                              (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, message2, (1, 60), cv2.FONT_HERSHEY_PLAIN,1,
                                              (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, message3, (1, 90), cv2.FONT_HERSHEY_PLAIN,1,
                                              (0, 255, 255), 1, cv2.LINE_AA)
            cv2.namedWindow('Frame',cv2.WND_PROP_FULLSCREEN,)
            cv2.setWindowProperty('Frame',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Frame',frame)

            if cv2.waitKey(1) & 0xFF == 27: #ESC鍵跳出
                thread_Flag1=False
                break
    cap.release()
    cv2.destroyAllWindows()

def Run_BtnAndLed():
    global thread_Flag1
    led.color=(0,1,0) #亮綠色
    while (thread_Flag1):
        if button.is_pressed:
                led.color=(1,0,0)
                #print("Button is pressed")
        else:
                led.color=(0,1,0)
                #print("Button is not pressed")
    led.color=(0,0,0)

if __name__ == '__main__':
    thread_Mediapipe = threading.Thread(target=Run_Mediapipe)  # 執行緒1:Mediapipe
    thread_BtnLed = threading.Thread(target=Run_BtnAndLed)     # 執行緒2:按鈕與LED控制
    thread_Mediapipe.start()
    thread_BtnLed.start()
