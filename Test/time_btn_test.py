import tkinter
from gpiozero import Button, RGBLED
import threading
import time

led=RGBLED(red=26,green=19,blue=13)
button = Button(12)

Flag1=True
Flag2=False
count1=0

def aa(): #這執行去給gpiozero用的
    global Flag1
    global count1
    led.color=(0,1,0)
    while (Flag1):
        button.wait_for_release()

    led.color=(0,0,0)

def btn_release():
    global Flag2
    Flag2=(not Flag2)
    if Flag2==True:
        led.color=(1,0,0)
    else:
        led.color=(0,1,0)
    time.sleep(0.2)

def bb(): #這執行續給TK用的
    global Flag1
    gui = tkinter.Tk()
    gui.title("Delftstack")
    # Updates activities
    gui.mainloop()
    Flag1=False

button.when_released = btn_release
a = threading.Thread(target=aa)  # 建立新的執行緒a
b = threading.Thread(target=bb)  # 建立新的執行緒b

a.start()  # 啟用a執行緒
b.start()  # 啟用b執行緒
