#imports
from re import template
import cv2
import pyautogui
from time import sleep
import numpy as np

#No cooldown time
pyautogui.PAUSE = 0

#template and dimensions
template = cv2.imread("new_nose.png")
template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
template_w, template_h = template_gray.shape[::-1]
# template_h, template_w = template_gray.shape

# game window dimensions (NOTE: THESE DIFFER BY SCREEN RESOLUTION!)
x, y, w, h = 1000, 250, 1500, 1000

#wait 3 seconds so we can set up the right screen
sleep(3)

#main loop
while True:

    #taking a screenshot and reading it in
    # temp_image = pyautogui.screenshot(region= (x, y, w, h))
    # image = cv2.cvtColor(np.array(temp_image),cv2.COLOR_RGB2BGR)
    pyautogui.screenshot("temp_image.png", (x, y, w, h))
    image = cv2.imread("temp_image.png")
    
    
    #sub loop - for template matching (finding max_val over and over again)
    while True:

        # show what the computer sees
        image_mini = cv2.resize(
            src = image,
            dsize = (450,350) #must be integer, not float
        )
        cv2.imshow("vision", image_mini)
        cv2.waitKey(10)
        
        #convert image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        #get matches
        result = cv2.matchTemplate(
            image = image_gray,
            templ = template_gray,
            method = cv2.TM_CCOEFF_NORMED
        )

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        #process greatest match if above threshold
        if max_val >= 0.8:
            # pyautogui.click(
            #    x = max_loc[0] + x, #screen x
            #    y = max_loc[1] + y  #screen y
            # )
            
            #draw over the matches, so we can match the next highest match
            image = cv2.rectangle(
                img = image,
                pt1 = max_loc,
                pt2 = (
                    max_loc[0] + template_w, # = pt2 x 
                    max_loc[1] + template_h # = pt2 y
                ),
                color = (0,0,255),
                thickness = -1 #fill the rectangle
            )
        
        else:
            break