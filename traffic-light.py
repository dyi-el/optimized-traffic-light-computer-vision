#!/usr/bin/python
import Adafruit_BBIO.GPIO as GPIO
import os
import time

os.system("config-pin P8_13 gpio")
os.system("config-pin P8_19 gpio")
os.system("config-pin P9_14 gpio")
os.system("config-pin P9_16 gpio")
os.system("config-pin P9_41 gpio")
os.system("config-pin P9_24 gpio")
os.system("config-pin P9_26 gpio")

# Leds R Y G
leds = [
    "P8_11", "P8_12", "P8_13",
    "P8_14", "P8_15", "P8_16",
    "P8_17", "P8_18", "P8_19",
    "P9_14", "P9_16", "P8_26"
]
ped_red = "P9_24"
ped_green = "P9_26"

# Input Actuators
pedestrian = "P9_12"
nextlane = "P9_27"
warning = "P9_23"

# Manual Mode Pin
manual = "P9_41"

# Config OUTPUTS & Initial LOW all LED
for led in range(len(leds)):
    GPIO.setup(leds[led], GPIO.OUT)
    GPIO.output(leds[led], GPIO.LOW)
GPIO.setup(ped_red, GPIO.OUT)
GPIO.setup(ped_green, GPIO.OUT)
GPIO.output(ped_red, GPIO.LOW)
GPIO.output(ped_green, GPIO.LOW)


# Config INPUTS    
GPIO.setup(pedestrian, GPIO.IN, GPIO.PUD_DOWN)
GPIO.setup(nextlane, GPIO.IN, GPIO.PUD_DOWN)
GPIO.setup(warning, GPIO.IN, GPIO.PUD_DOWN)
GPIO.setup(manual, GPIO.IN, GPIO.PUD_DOWN)


#---------------------Functions-----------------------

def init_pedestrian():
    print("Pedestrian signal")
    pedestrian_phase()


def init_warning():
    print("Warning signal")
    warning_phase()


def init_red(duration):
    GPIO.output(leds[0], GPIO.HIGH)
    GPIO.output(leds[3], GPIO.HIGH)
    GPIO.output(leds[6], GPIO.HIGH)
    GPIO.output(leds[9], GPIO.HIGH)
    GPIO.output(ped_red, GPIO.HIGH)
    print(str(duration)," seconds RED")
    time.sleep(duration)
    

def pedestrian_phase():
    print("Starting Pedestrian Phase")
    GPIO.output(ped_red, GPIO.LOW)
    GPIO.output(ped_green, GPIO.LOW)
    while True:
        GPIO.output(ped_green, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(ped_green, GPIO.LOW)
        time.sleep(0.5)
        if GPIO.input(pedestrian) == 1:
            print("Ending Pedestrian Phase")
            for i in range(3):
                GPIO.output(ped_red, GPIO.HIGH)
                time.sleep(0.5)
                GPIO.output(ped_red, GPIO.LOW)
                time.sleep(0.5)
            GPIO.output(ped_red, GPIO.HIGH)
            time.sleep(2)
            print("End Pedestrian Phase")
            break
    

def warning_phase():
    print("Starting Warning Phase")
    GPIO.output(leds[0], GPIO.LOW)
    GPIO.output(leds[3], GPIO.LOW)
    GPIO.output(leds[6], GPIO.LOW)
    GPIO.output(leds[9], GPIO.LOW)
    GPIO.output(leds[2], GPIO.LOW)
    GPIO.output(leds[5], GPIO.LOW)
    GPIO.output(leds[8], GPIO.LOW)
    GPIO.output(leds[11], GPIO.LOW)
    GPIO.output(ped_red, GPIO.LOW)
    while True:
        GPIO.output(leds[1], GPIO.HIGH)
        GPIO.output(leds[4], GPIO.HIGH)
        GPIO.output(leds[7], GPIO.HIGH)
        GPIO.output(leds[10], GPIO.HIGH)
        GPIO.output(ped_red, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(leds[1], GPIO.LOW)
        GPIO.output(leds[4], GPIO.LOW)
        GPIO.output(leds[7], GPIO.LOW)
        GPIO.output(leds[10], GPIO.LOW)
        GPIO.output(ped_red, GPIO.LOW)
        time.sleep(0.5)
        if GPIO.input(warning) == 1:
            print("Ending Warning Phase")
            GPIO.output(leds[0], GPIO.HIGH)
            GPIO.output(leds[3], GPIO.HIGH)
            GPIO.output(leds[6], GPIO.HIGH)
            GPIO.output(leds[9], GPIO.HIGH)
            GPIO.output(ped_red, GPIO.HIGH)
            time.sleep(2)
            print("End Warning Phase")
            break

def warning_timed(duration):
    for i in range(duration):
        GPIO.output(leds[1], GPIO.HIGH)
        GPIO.output(leds[4], GPIO.HIGH)
        GPIO.output(leds[7], GPIO.HIGH)
        GPIO.output(leds[10], GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(leds[1], GPIO.LOW)
        GPIO.output(leds[4], GPIO.LOW)
        GPIO.output(leds[7], GPIO.LOW)
        GPIO.output(leds[10], GPIO.LOW)
        time.sleep(0.5)
        print(str(i+1)," seconds elapsed for warning")



#------------------Initial Setup----------------------

# Warning Signal for 30s
warning_timed(5)

# Start RED for 5s
init_red(3)

fixed_lane_time = 5
fixed_ped_time = 7
lanes = 4
current_lane = 0




while True:
    # Manual Actuation Mode 
    print("Check Manual Mode")
    while GPIO.input(manual) == 1:
        if GPIO.input(nextlane) == 1:
            print("Currently on Manual Mode")
            
            current_lane = (current_lane + 1) % 4
            print("Next Lane ", str(current_lane+1)," signal")
            red = current_lane * (lanes-1)
            green = current_lane * (lanes-1) + 2
            
            prev_red = ((current_lane+3)%4) * (lanes-1)
            prev_yellow = ((current_lane+3)%4) * (lanes-1) + 1
            prev_green = ((current_lane+3)%4) * (lanes-1) + 2
            
            GPIO.output(leds[prev_red], GPIO.LOW)
            GPIO.output(leds[prev_green], GPIO.LOW)
            GPIO.output(leds[prev_yellow], GPIO.HIGH)
            time.sleep(3)
            GPIO.output(leds[prev_yellow], GPIO.LOW)
            GPIO.output(leds[prev_red], GPIO.HIGH)
            
            
            GPIO.output(leds[red], GPIO.LOW)
            GPIO.output(leds[green], GPIO.HIGH)
            
            
        if GPIO.input(pedestrian) == 1:
            print("Currently on Manual Mode")
            
            current_lane = (current_lane + 1) % 4
            red = current_lane * (lanes-1)
            green = current_lane * (lanes-1) + 2
            
            prev_red = ((current_lane+3)%4) * (lanes-1)
            prev_yellow = ((current_lane+3)%4) * (lanes-1) + 1
            prev_green = ((current_lane+3)%4) * (lanes-1) + 2
            
            GPIO.output(leds[prev_red], GPIO.LOW)
            GPIO.output(leds[prev_green], GPIO.LOW)
            GPIO.output(leds[prev_yellow], GPIO.HIGH)
            time.sleep(3)
            GPIO.output(leds[prev_yellow], GPIO.LOW)
            GPIO.output(leds[prev_red], GPIO.HIGH)
            init_pedestrian()
            
        if GPIO.input(warning) == 1:
            init_warning()
    
       
    # Automatic Actuation Mode         
    print("Currently on Automatic Mode")
    print("Lane ", str(current_lane+1) ," on GO")
    red = current_lane * (lanes-1)
    yellow = current_lane * (lanes-1) + 1
    green = current_lane * (lanes-1) + 2
    
    GPIO.output(leds[red], GPIO.LOW)
    GPIO.output(leds[green], GPIO.HIGH)
    time.sleep(fixed_lane_time)
    GPIO.output(leds[green], GPIO.LOW)
    
    GPIO.output(leds[yellow], GPIO.HIGH)
    time.sleep(3)
    GPIO.output(leds[yellow], GPIO.LOW)
    
    GPIO.output(leds[red], GPIO.HIGH)
    current_lane = (current_lane + 1) % 4
    

    
    
