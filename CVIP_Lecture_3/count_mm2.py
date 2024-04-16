import cv2
import numpy as np

def display_green_channel(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Check if image is loaded successfully
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Extract green channel
    green_channel = img[:, :, 1]  # Green channel is the second channel in OpenCV (0-based index)
    
    # Apply adaptive thresholding to make the green channel more monochrome
    _, green_channel_thresh = cv2.threshold(green_channel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Display the green channel after thresholding
    cv2.imshow('Green Channel after Thresholding', green_channel_thresh)
    
    # Perform k-means clustering to find green smarties
    kmeans_green_smarties(green_channel_thresh)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def kmeans_green_smarties(green_channel_thresh):
    # Convert to grayscale
    green_gray = cv2.cvtColor(green_channel_thresh, cv2.COLOR_BGR2GRAY)
    
    # Reshape image into a 2D array of pixels
    reshaped_img = green_gray.reshape((-1, 1))
    
    # Convert to floating-point array
    reshaped_img = np.float32(reshaped_img)
    
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2  # Number of clusters (considering background and green smarties)
    _, labels, centers = cv2.kmeans(reshaped_img, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert center values to uint8
    centers = np.uint8(centers)
    
    # Find the index of the cluster corresponding to green Smarties
    green_cluster_idx = np.argmax(centers[:, 0])  # Green Smarties would have higher pixel values
    
    # Reshape labels to match the original image shape
    labels = labels.reshape((green_channel_thresh.shape[0], green_channel_thresh.shape[1]))
    
    # Create a mask for green Smarties cluster
    green_smarties_mask = (labels == green_cluster_idx).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(green_smarties_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours around green Smarties
    result_img = cv2.cvtColor(green_channel_thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow('Green Smarties', result_img)

# Define file path
image_path = r"C:\Users\Gebruiker\OneDrive - Hanzehogeschool Groningen\Minor\minor IA\CVIP2024\CVIP_Lecture_3\images\smarties1.jpeg"

# Display the green channel of the image and perform k-means clustering to find green smarties
display_green_channel(image_path)
