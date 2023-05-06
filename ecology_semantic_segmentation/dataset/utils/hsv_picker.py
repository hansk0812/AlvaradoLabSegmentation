import cv2

img_file = "ecology_semantic_segmentation/dataset/resources/palette.png"

img = cv2.imread(img_file)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.namedWindow("f")

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_pixel = hsv[y, x]
        print (hsv_pixel)

cv2.setMouseCallback('f', onMouse)

cv2.imshow('f', img)
cv2.waitKey()
