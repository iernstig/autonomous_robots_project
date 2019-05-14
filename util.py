#!/usr/bin/env python
import numpy
import cv2


def calcAimPoint(blueHits, yellowHits, oldAimPoint):
  alpha = 0.5
  blueCoords = (0, 0)
  yellowCoords = (0, 0)
  nrBlueHits = len(blueHits)
  nrYellowHits = len(yellowHits)
  if (nrBlueHits > 0 and nrYellowHits > 0):
    for i in range(nrBlueHits):
      x = blueHits[i][0]
      y = blueHits[i][1]
      if (y > blueCoords[1]):
        blueCoords = (x, y)
    for i in range(nrYellowHits):
      x = yellowHits[i][0]
      y = yellowHits[i][1]
      if (y > yellowCoords[1]):
        yellowCoords = (x, y)
    xCoord = (blueCoords[0] + yellowCoords[0])/2
    yCoord = (blueCoords[1] + yellowCoords[1])/2
  elif (nrBlueHits > 0):
    xCoord = 160
    yCoord = 55
  elif (nrYellowHits > 0):
    xCoord = 480
    yCoord = 55
  else:
    xCoord = 320
    yCoord = 55

  xCoord = int(alpha*xCoord + (1-alpha)*oldAimPoint[0])
  yCoord = int(alpha*yCoord + (1-alpha)*oldAimPoint[1])
  return (xCoord, yCoord)

def calcSteeringAngle(aimPoint, integralPart):
    K_p_left = 0.3
    K_p_right = 0.2
    K_i = 0
    xCoord = aimPoint[0]
    error = (320 - xCoord)/320.0
    integralPart += error
    if (error > 0):
     steeringAngle = K_p_left * error + K_i * integralPart
    else:
      steeringAngle = K_p_right * error + K_i * integralPart
    if (steeringAngle < -0.3):
        steeringAngle = -0.3
        integralPart = 0
    elif (steeringAngle > 0.3):
        steeringAngle = 0.3
        integralPart = 0
    return steeringAngle, integralPart

def findCones(blue_img, yellow_img, image, cid):
  cnts, hier = cv2.findContours(blue_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  blue_hits = []
  for c in cnts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    area = cv2.contourArea(c)
    if (area < 2000 and area > 50):
      if (cid == 253):
        #print("Area: " + str(area))
        cv2.drawContours(image, [c], 0, (255, 0, 0), 2)
        cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)
        cv2.putText(image, "center", (cX-10, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      blue_hits.append((cX, cY))
  
  cnts, hier = cv2.findContours(yellow_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  yellow_hits = []
  for c in cnts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    area = cv2.contourArea(c)
    if (area < 2000 and area > 50):
      if (cid == 253):
        #print("Area: " + str(area))
        cv2.drawContours(image, [c], 0, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)
        cv2.putText(image, "center", (cX-10, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      yellow_hits.append((cX, cY))
  return blue_hits, yellow_hits, image

def findCircles(gray_image, color_image):
  circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, 
                             param1=220, param2=22,minRadius=5,maxRadius=40)
  circle_data = None
  if (circles is not None):
    circles = numpy.uint16(numpy.around(circles))
    circle_data = circles[0,:]
    for i in circles[0,:]:
      color_image = cv2.circle(color_image, (i[0], i[1]), i[2],(0,255,0),2)
      color_image = cv2.circle(color_image,(i[0],i[1]), 2, (0,0,255),3)
  return color_image, circle_data

def filterHitsOnCar(hits, circle_data, distance_thres, image):
  if (circle_data is not None):
    hits_new = []
    for cone in hits:
      tooClose = False
      for i in range(circle_data.shape[0]):
        distance = numpy.sqrt((cone[0] - circle_data[i,0])**2 + (cone[1] - circle_data[i, 1])**2)
        if (distance < distance_thres):
          tooClose = True
          break
      if (tooClose == False):
        hits_new.append(cone)
      else:
        cv2.drawMarker(image, position=cone, color=(255,0,255), markerType=cv2.MARKER_CROSS, thickness = 3)
        print("Hit removed!")
    return hits_new, image
  else:
    return hits, image

def detectCarCanny(img):
    canny = cv2.Canny(img, 100, 200)
    contours, hier= cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approximations = []

    for contour in contours:
        perimiter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimiter, True)
        approximations.append(approx)
    
    for approx in approximations:
        canny = cv2.drawContours(canny, approx, -1 , (0, 255, 0), 3) 
        if len(approx) == 4: # if square
            canny = cv2.drawContours(canny, approx, -1 , (0, 255, 0), 3) 
    return cann
def detectCarCircles(img):
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  #---------- detect circles ----------   
  circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, 
                             param1=80, param2=22, minRadius=5, maxRadius=40)
  

  CIRCLE_DISTANCE_THRESHOLD = 100
  if circles is not None:
    if circles.any() != 0:
      circles = numpy.uint16(numpy.around(circles))
      circles = circles[0]
      
      for circle in circles:
        x, y = circle[0], circle[1]
        radius = circle[2]
        # draw the circles 
        img = cv2.circle(img,(x, y), radius, (0, 0, 255), 3)

        # find circle-clusters to detect the car
        if len(circles) > 3:
          cluster_count = 0
          for other_circle in circles:
            x_other, y_other = other_circle[0], other_circle[1]
            if abs(x - x_other) < CIRCLE_DISTANCE_THRESHOLD and abs(y - y_other) < CIRCLE_DISTANCE_THRESHOLD:
              cluster_count += 1
            if cluster_count > 3:
              # cv2.putText(img, "car-detected (x:{}, y:{})".format(x, y),
              #             (250, 200),
              #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
              print("found car at x:{}, y{}".format(x, y))
              if x > 300:
                pass # should stop the car here? 
              
  return img


def find_blue_cones(blue_img, img):
  pass
def detectOrangeCones(blue_img, yellow_img, img, cid):
  # ---------- analyze each contour by color ----------
  cnts, hier = cv2.findContours(blue_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  blue_hits = []
  for c in cnts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    area = cv2.contourArea(c)
    if (area < 2000 and area > 50):
      if (cid == 253):
        cv2.drawContours(image, [c], 0, (255, 0, 0), 2)
        cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)
        cv2.putText(image, "center", (cX-10, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      blue_hits.append((cX, cY))
  
  cnts, hier = cv2.findContours(yellow_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  yellow_hits = []
  for c in cnts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    area = cv2.contourArea(c)
    if (area < 2000 and area > 50):
      if (cid == 253):
        #print("Area: " + str(area))
        cv2.drawContours(image, [c], 0, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)
        cv2.putText(image, "center", (cX-10, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      yellow_hits.append((cX, cY))
  return blue_hits, yellow_hits, image
