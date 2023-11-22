import tkinter
from gpiozero import Button, RGBLED
import threading
import time

led=RGBLED(red=26,green=19,blue=13)
button = Button(12)
led.color=(0,1,0)
flag1=True
count1=0

def aa(): #這執行去給gpiozero用的
    global flag1
    global count1
    while (flag1):
        if button.is_pressed:
                led.color=(1,0,0)
                #print("Button is pressed")
        else:
                led.color=(0,1,0)
                #print("Button is not pressed")
    led.color=(0,0,0)

def bb(): #這執行續給TK用的
    global flag1
    gui = tkinter.Tk()
    gui.title("Delftstack")
    # Updates activities
    gui.mainloop()
    flag1=False

a = threading.Thread(target=aa)  # 建立新的執行緒a
b = threading.Thread(target=bb)  # 建立新的執行緒b

a.start()  # 啟用a執行緒
b.start()  # 啟用b執行緒
