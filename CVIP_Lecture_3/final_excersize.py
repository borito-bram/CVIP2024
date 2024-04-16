import cv2
import numpy as np

# Load the image
image = cv2.imread(r'C:\Users\Gebruiker\OneDrive - Hanzehogeschool Groningen\Minor\minor IA\CVIP2024\CVIP_Lecture_3\images\smarties1.jpeg')


# Define standard resolution (e.g., 800x600)
standard_resolution = (500, 428)

# Resize the image to standard resolution
image = cv2.resize(image, standard_resolution)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (9, 9), 0)

binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('M&Ms adaptiveThreshold', binary_image)

# Detect circles using HoughCircles
circles = cv2.HoughCircles(binary_image, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                           param1=54, param2=25, minRadius=5, maxRadius=45)

# Initialize a dictionary to store colors and count of circles
circle_colors = {'black': 0, 'yellow': 0, 'orange': 0, 'blue': 0, 'green': 0, 'red': 0}

# Draw circles on the original image and find color within each circle
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Draw circle
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        
        # Extract the region of interest (ROI) within the circle
        roi = image[(y-int(r/2)):(y+int(r/2)), (x-int(r/2)):(x+int(r/2))]
        
        # Convert ROI to HSV for better color segmentation
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for M&Ms
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([15, 255, 255])
        
        lower_blue = np.array([100, 100, 50])  # Adjusted lower bound for blue
        upper_blue = np.array([130, 255, 255])  # Adjusted upper bound for blue
        
        lower_green = np.array([40, 40, 40])  # Adjusted lower bound for green
        upper_green = np.array([80, 255, 255])  # Adjusted upper bound for green
        
        lower_red = np.array([0, 60, 60])  # Adjusted lower bound for red
        upper_red = np.array([10, 255, 255])  # Adjusted upper bound for red
        
        # Mask the image to get only the color of the M&M
        mask_black = cv2.inRange(hsv_roi, lower_black, upper_black)
        mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
        mask_orange = cv2.inRange(hsv_roi, lower_orange, upper_orange)
        mask_blue = cv2.inRange(hsv_roi, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
        mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
        
        # Count the number of non-zero pixels in each mask
        black_pixels = cv2.countNonZero(mask_black)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        orange_pixels = cv2.countNonZero(mask_orange)
        blue_pixels = cv2.countNonZero(mask_blue)
        green_pixels = cv2.countNonZero(mask_green)
        red_pixels = cv2.countNonZero(mask_red)
        
        # Choose the color with the most pixels
        max_pixels = max(black_pixels, yellow_pixels, orange_pixels, blue_pixels, green_pixels, red_pixels)
        
        # Assign color label based on the color with the most pixels
        if max_pixels == black_pixels:
            color = 'black'
            cv2.putText(image, color, (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        elif max_pixels == yellow_pixels:
            color = 'yellow'
            cv2.putText(image, color, (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        elif max_pixels == orange_pixels:
            color = 'orange'
            cv2.putText(image, color, (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        elif max_pixels == blue_pixels:
            color = 'blue'
            cv2.putText(image, color, (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        elif max_pixels == red_pixels:
            color = 'red'
            cv2.putText(image, color, (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            color = 'green'
            cv2.putText(image, color, (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Increment count for the color in the dictionary
        circle_colors[color] += 1

# Display the result
cv2.imshow('M&Ms with Contours and Labels', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the color of each circle and its count
for color, count in circle_colors.items():
    print(f"Color: {color}, Count: {count}")
