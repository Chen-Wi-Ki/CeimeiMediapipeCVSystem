from gpiozero import RGBLED
from time import sleep

led=RGBLED(red=26,green=19,blue=13)

led.red=1
sleep(1)
led.color=(0,0,0)

led.green=1
sleep(1)
led.color=(0,0,0)

led.blue=1
sleep(1)
led.color=(0,0,0)

led.color=(1,1,1)
sleep(1)

led.color=(1,1,0)
sleep(1)

led.color=(0,1,1)
sleep(1)

led.color=(1,0,1)
sleep(1)


