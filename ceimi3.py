import cv2
cap = cv2.VideoCapture(0)
#cap.setWindowProperty('None',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cap.set(3,320)
cap.set(4,480)
cap.set(5,5)

import pyautogui
#s_w,s_h = pyautogui.size()
pyautogui.moveTo(319,479)

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils #mediapipe圖形繪製API
mp_hands = mp.solutions.hands #mediapipe手部API

from gpiozero import Button, RGBLED
led = RGBLED(red=26,green=19,blue=13) #RGD LED設定
button = Button(12) #按鍵設定
button_Flag1=False #這也作為recTimeFlag

import threading
thread_Flag1 = True #用於判斷是否結束所有迴圈

import time ,datetime
show_Result_flag=False #顯示結果的Flag
recTime = 60 #sec
showResultTime=5 #sec
Message_Result_Fail='  Fail...'
Message_Result_Good='  Good'
Message_Result_Exel='Exellent!!!'
Exel_or_Fail = 0
Exel_or_Fail_countFlag=False
angRaw = False
Show321_flag=False
Message_Show321=' '

import numpy as np
a = 0
b = 0
c = 0
x = 0
xxx=0
y = 0
yyy=0
z = 0
zzz=0
xx= 0
yy= 0
zz= 0
cal_array=[]
'''#中文字體,暫時不需要
from PIL import ImageFont, ImageDraw, Image
font = ImageFont.truetype('TaipeiSansTCBeta-Regular.ttf', 5)      # 設定字型與文字大小
'''
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

def Cumulative_calculation(value):
    global cal_array
    total_difference = 0
    cal_array = cal_array + [value]
    if len(cal_array) == 5:
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

def btn_release():
    global button_Flag1, show_Result_flag, Exel_or_Fail, Exel_or_Fail_countFlag, Show321_flag, Message_Show321
    global a,b,c,x,y,z,xxx,yyy,zzz,angRaw,message_IntegralAng
    button_Flag1=(not button_Flag1)
    if button_Flag1==True:
        led.color=(1,0,0)
        Exel_or_Fail = 0
        a = 0
        b = 0
        c = 0
        xxx=0
        yyy=0
        zzz=0

        Show321_flag = True
        Message_Show321='  3'
        time.sleep(1)
        Message_Show321='  2'
        time.sleep(1)
        Message_Show321='  1'
        time.sleep(1)
        Message_Show321='Start'
        time.sleep(1)
        Message_Show321=' '
        Show321_flag=False

        Exel_or_Fail_countFlag = True
        message_IntegralAng='AngX,AngY,AngZ'
        angRaw = True
        for i in range(0,60,1):
            time.sleep(1)
            if i==45:
                led.color=(0.4,0,0)
            elif i==50:
                led.color=(0.25,0,0)
            elif i==55:
                led.color=(0.05,0,0)
        angRaw=False
        show_Result_flag = True
        Exel_or_Fail_countFlag = False

        if Exel_or_Fail >=5:
            for i in range(0,10,1):
                led.color=(0,0,0)
                time.sleep(0.3)
                led.color=(0,1,0)
                time.sleep(0.3)
        else:
            for i in range(0,10,1):
                led.color=(0,0,0)
                time.sleep(0.3)
                led.color=(1,0,0)
                time.sleep(0.3)
        led.color=(0,1,0)
        show_Result_flag = False
        button_Flag1=(not button_Flag1)
    time.sleep(0.1)
    now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    path = '/home/wiki/Documents/'+now+'.csv'
    f = open(path, 'w')
    f.write(message_IntegralAng)
    f.close()
    time.sleep(0.1)

#----------------#
#主呼叫由這裡開始#
#----------------#
def Run_Mediapipe():
    global thread_Flag1, angRaw, show_Result_flag, Exel_or_Fail, message_IntegralAng
    global Message_Result_Exel, Message_Result_Good, Message_Result_Fail
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
                       #frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                       # mp_drawing.DrawingSpec(color=(255,50,255), thickness=2, circle_radius=1),
                       # mp_drawing.DrawingSpec(color=(200,160,200), thickness=1, circle_radius=1)
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
                        
                        cv2.circle(frame, (int(landmark_2[0]*320),int(landmark_2[1]*480)), 5, (0, 0, 255), -1)
                        cv2.circle(frame, (int(landmark_3[0]*320),int(landmark_3[1]*480)), 5, (0, 0, 255), -1)
                        cv2.circle(frame, (int(landmark_5[0]*320),int(landmark_5[1]*480)), 5, (0, 0, 255), -1)
                        cv2.line  (frame, (int(landmark_2[0]*320),int(landmark_2[1]*480)),(int(landmark_3[0]*320),int(landmark_3[1]*480)),(200,160,200),2)
                        cv2.line  (frame, (int(landmark_2[0]*320),int(landmark_2[1]*480)),(int(landmark_5[0]*320),int(landmark_5[1]*480)),(200,160,200),2)
                        
                        global a,b,c,xx,yy,zz,xxx,yyy,zzz

                        x,y,z = angle_x_y_z(landmark_3, landmark_2, landmark_5)
                        xx = Cumulative_calculation(x)
                        yy = Cumulative_calculation(y)
                        zz = Cumulative_calculation(z)
                        if xx != 0:
                            xxx = xx
                        else:
                            xx = xxx
                        if yy != 0:
                            yyy = yy
                        else:
                            yy = yyy
                        if zz != 0:
                            zzz = zz
                        else:
                            zz = zzz

                        if Exel_or_Fail_countFlag==True:
                            if x > 1:
                                a = 1
                            else:
                                b = 1

                            if a == 0 and b == 1:
                                c = 1

                            if a == 1 and b == 0:
                                c = 2

                            if a == 1 and c == 1:
                                a = 0
                                b = 0
                                c = 0
                                Exel_or_Fail = Exel_or_Fail+1

                            if b == 1 and c == 2:
                                a = 0
                                b = 0
                                c = 0
                #左上顯示計算的手腕旋轉角度
                #message1 = 'AngX='+str(int(x))+','+str(int(xx))
                #message2 = 'AngY='+str(int(y))+','+str(int(yy))
                #message3 = 'AngZ='+str(int(z))+','+str(int(zz))

            #else:
                #message1 = 'AngX dissonance'
                #message2 = 'AngY dissonance'
                #message3 = 'AngZ dissonance'

            #cv2.putText(frame, message1, (1, 30), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 255), 1, cv2.LINE_AA)
            #cv2.putText(frame, message2, (1, 60), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 255), 1, cv2.LINE_AA)
            #cv2.putText(frame, message3, (1, 90), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 255), 1, cv2.LINE_AA)
            if angRaw==True: #開啟角度積分紀錄
                message_IntegralAng = message_IntegralAng+'\n'+str(int(xxx))+','+str(int(yyy))+','+str(int(zzz))
            if Show321_flag==True: #秀321倒數
                cv2.putText(frame, Message_Show321, (7, 320), cv2.FONT_HERSHEY_SIMPLEX,4,(0, 0, 255), 3, cv2.LINE_AA)
            if show_Result_flag==True: #秀評分
                if Exel_or_Fail>=6:
                    cv2.rectangle(frame, (0, 185), (360, 250), (255, 200, 0), cv2.FILLED)
                    cv2.putText(frame, Message_Result_Exel, (14, 240), cv2.FONT_HERSHEY_SIMPLEX,2,
                                                          (0, 255, 0), 3, cv2.LINE_AA)
                elif Exel_or_Fail>=5:
                    cv2.rectangle(frame, (0, 185), (360, 250), (255, 200, 0), cv2.FILLED)
                    cv2.putText(frame, Message_Result_Good, (14, 240), cv2.FONT_HERSHEY_SIMPLEX,2,
                                                          (0, 255, 255), 3, cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, (0, 185), (360, 250), (255, 200, 0), cv2.FILLED)
                    cv2.putText(frame, Message_Result_Fail, (14, 240), cv2.FONT_HERSHEY_SIMPLEX,2,
                                                          (0, 0, 255), 3, cv2.LINE_AA)

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
        button.wait_for_release()
    led.color=(0,0,0)

if __name__ == '__main__':
    button.when_released = btn_release
    thread_Mediapipe = threading.Thread(target=Run_Mediapipe)  # 執行緒1:Mediapipe
    thread_BtnLed = threading.Thread(target=Run_BtnAndLed)     # 執行緒2:按鈕與LED控制
    thread_Mediapipe.start()
    thread_BtnLed.start()
