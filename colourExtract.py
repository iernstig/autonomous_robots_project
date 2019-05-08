import cv2

print("Enter file name")
file = raw_input() 

while 0 == 0:  
  img = cv2.imread(file, 1)
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  print("Enter x-coord:")
  x = raw_input()
  if x == "exit":
    print("Exiting..")
    exit()
  x = int(x)
  print("Enter y-coord:")
  y = raw_input()
  if y == "exit":
    print("Exiting..")
    exit()

  y = int(y)
  
  print("H-value:" + str(hsv_img[y, x, 0]))
  print("S-value:" + str(hsv_img[y, x, 1]))
  print("V-value:" + str(hsv_img[y, x, 2]))
  img = cv2.drawMarker(img, position=(x,y), color=(0,0,255), markerType=cv2.MARKER_CROSS)

  cv2.imshow('img', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
