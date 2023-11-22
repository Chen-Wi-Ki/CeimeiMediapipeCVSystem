from gpiozero import Button, RGBLED
from time import sleep

led=RGBLED(red=26,green=19,blue=13)
button=Button(12)

#button.wait_for_press()
#print("Button was pressed")
led.color=(0,1,0)

while True:
	if button.is_pressed:
		led.color=(1,0,0)
		#print("Button is pressed")
	else:
		led.color=(0,1,0)
		#print("Button is not pressed")

led.color=(0,0,0)

