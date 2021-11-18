# import Adafruit_GPIO.SPI as SPI
from MCP3008 import MCP3008
import time
import os

adc = MCP3008()

file = open("ecgOut.txt","w")
while True:
    # Read all the ADC channel values in a list.
    value = adc.read(channel = 0)
    ################
    value = value / 1023.0 * 3.3
    ################
    file.write(str(value))
    file.write('\n')
    print(value)
    time.sleep(.004)
