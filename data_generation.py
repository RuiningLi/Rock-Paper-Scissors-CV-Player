# Copyright @2021 Ruining Li. All Rights Reserved.

from optparse import OptionParser
import os
import cv2
import keyboard
from PIL import Image

"""
This Python script is used to generate image data for different gestures in the game Rock-Paper-Scissors.
To generate data, please create a new folder named "dataset" at the same directory as this script file. Then, execute the 
command "sudo python3 data_generation.py" and follow the instruction to complete the generation process. 
The generation process will capture 100 images and save them in the folder "dataset/paper" or "dataset/scissors" or 
"dataset/rock", named from "0.png" to "99.png".

You can also add an optional argument at the end of the command to specify the starting name of the images. For example, 
"sudo python3 data_generation.py 200" will capture 100 images and save them in the corresponding folder, named from "200.png" 
to "299.png".
"""

def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()
    type_ = input("Which kind of gesture are you about to generate? Please choose one from \"paper\", \"scissors\", and \"rock\": ")
    if type_ != "paper" and type_ != "scissors" and type_ != "rock":
        print("INVALID! Please try again.")
        return
    path = "dataset/" + type_
    os.makedirs(path, exist_ok=True)
    print("Ready to generate data in dataset/" + type_ + " folder.")
    print("Press the Enter key to start taking pictures")
    print("or press the Esc key to quit.")

    cap = cv2.VideoCapture(0)
    start_taking_pictures = False
    num_images_taken = 0
    NUM_IMAGES_REQUIRED = 100
    while cap.isOpened() and num_images_taken < NUM_IMAGES_REQUIRED:
        if keyboard.is_pressed('esc'):
            break
        success, image = cap.read()
        if not success:
            continue
        cv2.imshow('Camera', image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not start_taking_pictures and keyboard.is_pressed('enter'):
            start_taking_pictures = True
        if start_taking_pictures:
            image_file = Image.fromarray(image)
            image_file.save(path + "/" + str(num_images_taken + (0 if len(args) == 0 else int(args[0]))) + ".png")
            num_images_taken += 1
        if cv2.waitKey(5) & 0xFF == 27:
            break

    print(str(num_images_taken), "images saved!")
    cap.release()

if __name__ == "__main__":
    main()
