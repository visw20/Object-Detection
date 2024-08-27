# import cv2
# import numpy as np

# #paths to the configuration, weights, and class names files
# config_path = "C:/Users/Viswajith/Downloads/Unconfirmed 915762.crdownload"
# weights_path = "C:/Users/Viswajith/Downloads/yolov4-tiny.weights"
# names_path = "C:/Users/Viswajith/Downloads/coco.names"

# #load yolo
# net = cv2.dnn.readNet(weights_path, config_path)
# layer_names = net.getLayerNames()
# output_layer = [layer_names[i - 1] for i in net.getUnconnectedOutLayer()]

# #load class names
# classes = []
# with open(names_path, "r") as f:
#     classes= [line.strip() for line in f.readlines()]
    
# #generate random colors for each class
# np.random.seed(42)
# colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

# #load image
# image = cv2.imread("C:/Users/Viswajith/Downloads/car person bus.webp")

# #Resize image for better quality
# scale_precent = 150 #percent of original size
# width = int(image.shape[1] * scale_percent / 100)
# height = int(image.shape[0] * scale_percent /100)
# dim = (width, height)
# image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# #detecing objects
# blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
# net.setInput(blob)
# outs = net.forward(output_layers)

# #showing information on the screen
# class_ids = []
# confidences = []
# boxes = []

# for out in outs:
#     for detection in out:
#         scores = detection[5:]
#         class_id = np.argmax(scores)
#         confidence = score[class_id]
#         if confidence > 0.1: #adjust confidence threshold here
#             #object detection
#             center_x = int(detection[0] * width)
#             center_y = int(detection[1] * height)
#             w = int(detection[2] * width)
#             h = int(detection[3] * height)
            
#             #rectangle coordinates
#             x = int(center_x - w / 2)
#             y = int(center_y - h / 2)
            
#             boxes.append([x, y, w, h])
#             confidences.append(float(confidence))
#             class_ids.append(class_id)
            
# #apply non-maxima suppression
# indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.1, nms_threshold=0.4)

# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# font_thickness = 2 

# #track courts of detected objects
# class_court = {class_name: 0 for class_name in classes}

# for i in range(len(boxes)):
#     if i in indexes:
#         x, y, w, h = boxes[i]
#         label = str(classes[class_ids[i]])
#         color = [int(c) for c in colors[class_ids[i]]]
        
#         #Draw rectangle and label
#         cv2.rectangle(image, (x, y), (x + w, y+ h),color, 3)
#         text = f"{label} {confidences[i]*100:.2f}"
#         cv2.putText(image, text, (x, y - 10), font, font_scale, color, font_thickness)
        
#         #count the number of each class detected
#         class_court[label] += 1

# #display the court of each detected object class in the top-right corner
# text_color = (255, 255, 255)
# text_offset_y =30
# text_offset_x = image.shape[1] - 250 #adjust this value based on the image size and text length
# text_line_height = 30
# for i, class_name in enumerate(["person", "car", "bus"]):
#     court_text = f"{class_name}: {class_court[class_name]}"
#     cv2.putText(image, court_text, (text_offset_x, text_offset_y + i * text_line_height),font, 1, text_color, 2)
    
# #set up the display window to be full screen
# cv2.nameWindow("Image", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# #Display the image
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






import cv2
import numpy as np
import os

# Paths to the configuration, weights, and class names files
config_path = "C:/Object Dectection/yolov4-tiny.cfg.txt"  # Ensure this file exists
weights_path = "C:/Object Dectection/yolov4-tiny.weights"  # Ensure this file exists
names_path = "C:/Object Dectection/coco.names"  # Ensure this file exists

# Verify the paths
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights file not found: {weights_path}")
if not os.path.exists(names_path):
    raise FileNotFoundError(f"Names file not found: {names_path}")

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
classes = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Generate random colors for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

# Load image
image_path = "C:/Users/Viswajith/Downloads/download (1).jpg"
image = cv2.imread(image_path)

# Check if image is loaded successfully
if image is None:
    raise FileNotFoundError(f"Image file not found or cannot be read: {image_path}")

# Resize image for better quality
scale_percent = 150  # Percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Detecting objects
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing information on the screen
class_ids = []
confidences = []
boxes = []

height, width, _ = image.shape

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.1:  # Adjust confidence threshold here
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maxima suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

# Track counts of detected objects
class_counts = {class_name: 0 for class_name in classes}

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = [int(c) for c in colors[class_ids[i]]]

        # Draw rectangle and label
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        text = f"{label} {confidences[i] * 100:.2f}"
        cv2.putText(image, text, (x, y - 10), font, font_scale, color, font_thickness)

        # Count the number of each class detected
        class_counts[label] += 1

# Display the count of each detected object class in the top-right corner
text_color = (255, 255, 255)
text_offset_y = 30
text_offset_x = image.shape[1] - 250  # Adjust this value based on the image size and text length
text_line_height = 30
for i, class_name in enumerate(["person", "car", "bus"]):
    count_text = f"{class_name}: {class_counts[class_name]}"
    cv2.putText(image, count_text, (text_offset_x, text_offset_y + i * text_line_height), font, 1, text_color, 2)

# Set up the display window to be full screen
cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()










































            
            
        






























