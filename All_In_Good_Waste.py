#  -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation.  All rights reserved.
#  -------------------------------------------------------------
"""
Skeleton code showing how to load and run the TensorFlow Lite export package from Lobe.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

EXPORT_MODEL_VERSION = 1

class TFLiteModel:
    def __init__(self, model_dir) -> None:
        """Method to get name of model file. Assumes model is in the parent directory for script."""
        with open(os.path.join(model_dir, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.model_file = "../" + self.signature.get("filename")
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"Model file does not exist")
        self.interpreter = None
        self.signature_inputs = self.signature.get("inputs")
        self.signature_outputs = self.signature.get("outputs")
        # Look for the version in signature file.
        # If it's not found or the doesn't match expected, print a message
        version = self.signature.get("export_model_version")
        if version is None or version != EXPORT_MODEL_VERSION:
            print(
                f"There has been a change to the model format. Please use a model with a signature 'export_model_version' that matches {EXPORT_MODEL_VERSION}."
            )

    def load(self) -> None:
        """Load the model from path to model file"""
        # Load TFLite model and allocate tensors.
        self.interpreter = tflite.Interpreter(model_path=self.model_file)
        self.interpreter.allocate_tensors()
        # Combine the information about the inputs and outputs from the signature.json file with the Interpreter runtime
        input_details = {detail.get("name"): detail for detail in self.interpreter.get_input_details()}
        self.model_inputs = {key: {**sig, **input_details.get(sig.get("name"))} for key, sig in self.signature_inputs.items()}
        output_details = {detail.get("name"): detail for detail in self.interpreter.get_output_details()}
        self.model_outputs = {key: {**sig, **output_details.get(sig.get("name"))} for key, sig in self.signature_outputs.items()}
        if "Image" not in self.model_inputs:
            raise ValueError("Tensorflow Lite model doesn't have 'Image' input! Check signature.json, and please report issue to Lobe.")

    def predict(self, image) -> dict:
        """
        Predict with the TFLite interpreter!
        """
        if self.interpreter is None:
            self.load()

        # process image to be compatible with the model
        input_data = self.process_image(image, self.model_inputs.get("Image").get("shape"))
        # set the input to run
        self.interpreter.set_tensor(self.model_inputs.get("Image").get("index"), input_data)
        self.interpreter.invoke()

        # grab our desired outputs from the interpreter!
        # un-batch since we ran an image with batch size of 1, and convert to normal python types with tolist()
        outputs = {key: self.interpreter.get_tensor(value.get("index")).tolist()[0] for key, value in self.model_outputs.items()}
        return self.process_output(outputs)

    def process_image(self, image, input_shape) -> np.ndarray:
        """
        Given a PIL Image, center square crop and resize to fit the expected model input, and convert from [0,255] to [0,1] values.
        """
        width, height = image.size
        # ensure image type is compatible with model and convert if not
        if image.mode != "RGB":
            image = image.convert("RGB")
        # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            # Crop the center of the image
            image = image.crop((left, top, right, bottom))
        # now the image is square, resize it to be the right shape for the model input
        input_width, input_height = input_shape[1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))

        # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
        image = np.asarray(image) / 255.0
        # format input as model expects
        return image.reshape(input_shape).astype(np.float32)

    def process_output(self, outputs) -> dict:
        # postprocessing! convert any byte strings to normal strings with .decode()
        out_keys = ["label", "confidence"]
        for key, val in outputs.items():
            if isinstance(val, bytes):
                outputs[key] = val.decode()

        # get list of confidences from prediction
        confs = list(outputs.values())[0]
        labels = self.signature.get("classes").get("Label")
        output = [dict(zip(out_keys, group)) for group in zip(labels, confs)]
        sorted_output = {"predictions": sorted(output, key=lambda k: k["confidence"], reverse=True)}
        return sorted_output

#https://www.digikey.ca/en/maker/projects/make-a-pi-trash-classifier-with-machine-learning-and-lobe/9029e053c23845e98bea70c5cabfb555
# ------------------------------------------------------------------------
# Trash Classifier ML Project
# Please review ReadMe for instructions on how to build and run the program
# (c) 2020 by Jen Fox, Microsoft
# MIT License
# --------------------------------------------------------------------------
#import Pi GPIO library button class
from gpiozero import Button, LED, PWMLED
from picamera import PiCamera
from time import sleep
from lobe import ImageModel

#Create input, output, and camera objects
button = Button(26) #MF*****was 2
white_led = PWMLED(6) #ready light
yellow_led = LED(17) #cardboard
blue_led = LED(27) #glass
green_led = LED(22) #metal
red_led = LED(5) #paper
orange_led = LED(13) #plastic
new_led = LED(19) #trash
camera = PiCamera()

# Load Lobe TF model
# --> Change model file path as needed
####model = ImageModel.load('/home/pi/Lobe/model/Garbage classification TFLite')
# Take Photo
def take_photo():
    # Quickly blink status light
    white_led.blink(0.1,0.1)
    sleep(2)
    print("Camera Pressed")
    white_led.on()
    # Start the camera preview
    camera.start_preview(alpha=200)
    # wait 2s or more for light adjustment
    sleep(3)
    # Optional image rotation for camera
    # --> Change or comment out as needed
    camera.rotation = 270
    #Input image file path here
    # --> Change image path as needed
    camera.capture('/home/pi/Pictures/image.jpg')
    #Stop camera
    camera.stop_preview()
    white_led.off()
    sleep(1)

# Identify prediction and turn on appropriate LED
def led_select(label):
    print(label)
    if label == "cardboard":
        yellow_led.on()
        sleep(5)
    if label == "glass":
        blue_led.on()
        sleep(5)
    if label == "metal":
        green_led.on()
        sleep(5)
    if label == "paper":
        red_led.on()
        sleep(5)
    if label == "plastic":
        new_led.on() #white_led.on()
        sleep(5)
    if label == "trash":
        orange_led.on()
        sleep(5)
        orange_led.off()  #MF Added this code as it was not turning off for some reason.
    else:
        yellow_led.off()
        blue_led.off()
        green_led.off()
        red_led.off()
        white_led.off()
        orange_led.off()
        new_led.off()

#https://www.hackster.io/mjrobot/playing-with-electronics-rpi-gpio-zero-library-tutorial-f984c9
import time
import sys
from gpiozero import OutputDevice as stepper
IN1 = stepper(14)
IN2 = stepper(15)
IN3 = stepper(20)
IN4 = stepper(21)
IN5 = stepper(24, True)  #enable
IN6 = stepper(23, True)  #enable
stepPins = [IN1,IN2,IN3,IN4] # Motor GPIO pins</p><p>
stepDir = 1        # Set to 1 for clockwise
                           # Set to -1 for anti-clockwise
mode = 1            # mode = 1: Low Speed ==> Higher Power
                           # mode = 0: High Speed ==> Lower Power
if mode:              # Low Speed ==> High Power
  seq = [[1,0,0,1], # Define step sequence as shown in manufacturers datasheet
             [1,0,0,0],
             [1,1,0,0],
             [0,1,0,0],
             [0,1,1,0],
             [0,0,1,0],
             [0,0,1,1],
             [0,0,0,1]]
else:                    # High Speed ==> Low Power
  seq = [[1,0,0,0], # Define step sequence as shown in manufacturers datasheet
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]]
stepCount = len(seq)
if len(sys.argv)>1: # Read wait time from command line
  waitTime = 0.002 #int(sys.argv[1])/float(1000)
else:
  waitTime = 0.002 #0.004    # 2 miliseconds was the maximun speed got on my tests
stepCounter = 0

#!/usr/bin/env python3
########################################################################
# Filename    : I2CLCD1602.py
# Description : Use the LCD display data
# Author      : freenove
# modification: 2018/08/03
########################################################################
from PCF8574 import PCF8574_GPIO
from Adafruit_LCD1602 import Adafruit_CharLCD
#MF MUST COPY SUPPORTING FILES FROM FREENOVE TUTRIAL 20
from time import sleep, strftime
from datetime import datetime
 
def get_cpu_temp():     # get CPU temperature and store it into file "/sys/class/thermal/thermal_zone0/temp"
    tmp = open('/sys/class/thermal/thermal_zone0/temp')
    cpu = tmp.read()
    tmp.close()
    return '{:.2f}'.format( float(cpu)/1000 ) + ' C'
 
def get_time_now():     # get system time
    return datetime.now().strftime('    %H:%M:%S')
   
def loop():
    mcp.output(3,1)     # turn on LCD backlight
    lcd.begin(16,2)     # set number of LCD lines and columns
    while(True):        
        #lcd.clear()
        lcd.setCursor(0,0)  # set cursor position
        lcd.message( 'CPUx: ' + get_cpu_temp()+'\n' )# display CPU temperature
        lcd.message( get_time_now() )   # display the time
        sleep(1)
       
def destroy():
    lcd.clear()
   
PCF8574_address = 0x27  # I2C address of the PCF8574 chip.
PCF8574A_address = 0x3F  # I2C address of the PCF8574A chip.
# Create PCF8574 GPIO adapter.
try:
    mcp = PCF8574_GPIO(PCF8574_address)
except:
    try:
        mcp = PCF8574_GPIO(PCF8574A_address)
    except:
        print ('I2C Address Error !')
        #exit(1) #MF added
# Create LCD, passing in MCP GPIO adapter.
lcd = Adafruit_CharLCD(pin_rs=0, pin_e=2, pins_db=[4,5,6,7], GPIO=mcp)

def motormf(stepCounter,stepDir,angle):
            print("motormf called")
            for i in range(0,angle): #MF Control angle
             for pin in range(0,4):
               xPin=stepPins[pin]          # Get GPIO
               if seq[stepCounter][pin]!=0:
                 xPin.on()
               else:
                 xPin.off()
             stepCounter += stepDir
             if (stepCounter >= stepCount):
               stepCounter = 0
             if (stepCounter < 0):
               stepCounter = stepCount+stepDir
             #print(stepCounter)
             time.sleep(waitTime)     # Wait before moving on

# Main Function
while True:
    if button.is_pressed:
        take_photo()
        # Run photo through Lobe TF model
        ####result = model.predict_from_file('/home/pi/Pictures/image.jpg')
        # --> Change image path
        #led_select(result.prediction)      
       
#if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Predict a label for an image.")
    #parser.add_argument("image", help="Path to your image file.")
    #args = parser.parse_args()
    # Assume model is in the parent directory for this file
        model_dir = os.path.join(os.getcwd(), "..")
        #data_folder = os.path.join("/home","pi","Pictures","trash7.jpg")
        data_folder = os.path.join("/home","pi","Pictures","image.jpg")
        print(data_folder)
    #image2 = open(data_folder,"r")

    #if os.path.isfile(image):
        image = Image.open(data_folder)
        model = TFLiteModel(model_dir)
        model.load()
        outputs = model.predict(image)
        print(f"Predicted: {outputs}")
    #else:
        print("-------")
        print(outputs.values())
        print("-------")
        print(outputs.items())
        print("-------")
        for key,value in outputs.items():
            print(key)
            print('\n')
            print(value)
            print('\n')
            #for key, value in key.items():
                #print(x, y)
            for i in value:
                print(i)
                print("-------")
                print (value[0])
                print("-------")
                for k,v in value[0].items():
                    if type(v) != float:
                        print(v)
                        category = v
                    if type(v) == float:
                        print(v)
                        confidence = str(v*100) #MF TypeError: can only concatenate str (not "float") to str    
                        confidenceStr = "{:10.1f}".format(v*100).strip() + "%"                  
            print("Answer is: " + category + " you idiot!!!")
            print("Answer is: " + confidence + " you idiot!!!")
            print("Answer is: " + confidenceStr + " you idiot!!!")
           
#if __name__ == '__main__':
        print ('Program is starting LCD... ')
        try:
            #loop()
            mcp.output(3,1)     # turn on LCD backlight
            lcd.begin(20,4) #(16,2)     # set number of LCD lines and columns

            lcd.clear()
            #lcd.message( '12345678901234567890' + '\n')
            lcd.setCursor(0,0)  # set cursor position
            lcd.message( 'LOBE.AI conf: ' + confidenceStr + '\n')
            lcd.setCursor(0,1)  # set cursor position
            lcd.message( 'Category: ' + category + '\n')          
            lcd.setCursor(0,2)  # set cursor position
            if (category == "cardboard" or category == "paper"):                
                lcd.message('Bin A' + ' - turn left' + '\n')
            elif (category == "glass" or category == "metal" or category == "plastic"):
                lcd.message('Bin B' + ' - turn left' + '\n')
            elif (category == "trash"):
                lcd.message('Bin C' + ' - turn right' + '\n')
            else:
                lcd.message('Error reading picture' + '\n')    
            lcd.setCursor(0,3)  # set cursor position
            lcd.message('Pick Up Your Trash!' + '\n')          
            sleep(1)
            print("lcd...")
           
        except KeyboardInterrupt:
            destroy()        
       
        led_select(category)
        print("category sent")
        led_select("cardboard")
        led_select("glass")
        led_select("metal")
        led_select("paper")
        led_select("plastic")
        led_select("trash")
       
#if __name__ == '__main__':     # Program entrance
        print ('Motor Program is starting...')
        try:
        # Start main loop
        #while True:
          photo = category
          if (category == "cardboard" or category == "paper"):
            print ('Left track')  
            IN5.on() # = stepper(23, True)  #enable
            IN6.off() # = stepper(24, True)  #enable            
            motormf(stepCounter,1,1600*2)
            print ('Left tray')
            sleep(1)
            IN5.off() # = stepper(23, True)  #enable
            IN6.on() # = stepper(24, True)  #enable
            motormf(stepCounter,1,800)
            motormf(stepCounter,-1,800*2)
            IN5.on() # = stepper(23, True)  #enable
            IN6.off() # = stepper(24, True)  #enable
            motormf(stepCounter,-1,1600*2*2)
          elif (category == "glass" or category == "metal" or category == "plastic"):
            print ('Left track')  
            IN5.on() # = stepper(23, True)  #enable
            IN6.off() # = stepper(24, True)  #enable            
            motormf(stepCounter,1,1600*2)
            print ('Left tray')
            sleep(1)
            IN5.off() # = stepper(23, True)  #enable
            IN6.on() # = stepper(24, True)  #enable
            motormf(stepCounter,1,800)
            motormf(stepCounter,-1,800*2)
            IN5.on() # = stepper(23, True)  #enable
            IN6.off() # = stepper(24, True)  #enable
            motormf(stepCounter,-1,1600*2*2)
          elif(category == "trash"):
            print ('left track *2')  
            IN5.on() # = stepper(23, True)  #enable
            IN6.off() # = stepper(24, True)  #enable            
            motormf(stepCounter,1,1600*4)
            print ('Left tray')
            sleep(1)
            IN5.off() # = stepper(23, True)  #enable
            IN6.on() # = stepper(24, True)  #enable
            motormf(stepCounter,-1,800*2)
            motormf(stepCounter,1,800)
            IN5.on() # = stepper(23, True)  #enable
            IN6.off() # = stepper(24, True)  #enable
            motormf(stepCounter,-1,1600*2*4)
          else:
            print('Error reading picture')    

          print ('Motor Loop ended.')
         
        except KeyboardInterrupt:  # Press ctrl-c to end the program.
            destroy()
 
    else:
        # Pulse status light
        white_led.pulse(2,1)
    sleep(1)
