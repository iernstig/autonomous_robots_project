#!/usr/bin/env python
import numpy
import cv2


def calcAimPoint(blueHits, yellowHits, oldAimPoint):
  '''
  Calculating aimpoint based on yellow and blue cones using
  exponential moving avarage for a smoother effect. The aimpoint
  is put in the middle of the blue and yellow cones that are furthest
  down in the image (closest to the car). If only blue or yellow or 
  no cones at all are found the aimpoint is put at predefined positions.
  '''
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


def calcOrangeAimPoint(orangeHits, oldAimPoint, img):
  '''
  Calculating aimpoint only based on orange cones. The aimpoint
  is put in the middle of the two orange cones on left and right
  side that are furthest down.
  '''
  alpha = 0.5
  leftCoords = (0,0)
  rightCoords = (639,0)
  for hit in orangeHits:
    # Find the cone on the left that is furthest down
    if (hit[0] < 320):
      if (hit[1] > leftCoords[1]):
        leftCoords = hit
    # Find the cone on the right that is furthest down
    else:
      if (hit[1] > rightCoords[1]):
        rightCoords = hit
  if(leftCoords != (0,0) and rightCoords != (639,0)):
    img = cv2.drawMarker(img, position=leftCoords, color=(0,255,255), markerType=cv2.MARKER_CROSS, thickness=3)
    img = cv2.drawMarker(img, position=rightCoords, color=(255,255,0), markerType=cv2.MARKER_CROSS, thickness=3)
    xCoord = (leftCoords[0] + rightCoords[0])/2
    yCoord = (leftCoords[1] + rightCoords[1])/2
  # If cones are not found in both left and right section of the image the aimpoint is put in the middle of the image.
  else:
    xCoord = 320
    yCoord = 55
  
  # Exponential moving average.
  xCoord = int(alpha*xCoord + (1-alpha)*oldAimPoint[0])
  yCoord = int(alpha*yCoord + (1-alpha)*oldAimPoint[1])
  return (xCoord, yCoord), img


def calcSteeringAngle(aimPoint, integralPart):
    '''
    Calculating steering angle based on aimpoint, implemented as a PI-regulator,
    however the integral part has been put to zero. The regulator tries to keep the
    x-coordinate of the aimpoint in the middle of the image.
    '''
    # Different gains for left and right steering since the car has a bias.
    K_p_left = 0.3
    K_p_right = 0.2
    K_i = 0
 
    xCoord = aimPoint[0]
    error = (320 - xCoord)/320.0
    integralPart += error
    # the case when we want to turn left
    if (error > 0):
     steeringAngle = K_p_left * error + K_i * integralPart
    # the case when we want to turn right
    else:
      steeringAngle = K_p_right * error + K_i * integralPart
    # Keep the steering angle within allowd bounds.
    if (steeringAngle < -0.3):
        steeringAngle = -0.3
        integralPart = 0
    elif (steeringAngle > 0.3):
        steeringAngle = 0.3
        integralPart = 0
    return steeringAngle, integralPart


def findCones(color_filtered_img, image, cid, color):
  '''
  Find cones from colorfiltered binary images, using openCVs findContours. Hits are filtered out
  if they are too small or large to avoid some false positives. The contours are plotted if in
  replay mode.
  '''
  im2, cnts, hier = cv2.findContours(color_filtered_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  hits = []
  for c in cnts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    area = cv2.contourArea(c)
    if (area < 2000 and area > 50):
      if (cid == 253):
        cv2.drawContours(image, [c], 0, color, 2)
        cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)
        cv2.putText(image, "center", (cX-10, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      hits.append((cX, cY))
  
  return hits, image


def findCircles(gray_image, color_image):
  '''
  Find circles in a black and white image, used to remove false postive cones
  that lie on another car. 
  '''
  circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, 
                             param1=220, param2=22,minRadius=5,maxRadius=40)
  circle_data = None
  if (circles is not None):
    circles = numpy.uint16(numpy.around(circles))
    circle_data = circles[0,:]
  return color_image, circle_data


def filterHitsOnCar(hits, circle_data, distance_thres, image):
  '''
  Remove false postive cones. This is done by removing hits that are closer than
  dist_thers to the center of a circle. 
  '''
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
        print("Hit removed!")
    return hits_new, image
  else:
    return hits, image


def detectCarCircles(gray_image, img):
  '''
  Find position of another car based on Hough circle transform. This is done
  by finding clusters of circles that are closer than CIRCLE_DISTANCE_THERSHOLD.
  If the cluster consists of at least 4 circles it is assumed to be a car and
  x,y coordinates are returned.
  '''
  
  #---------- detect circles ----------   
  circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 15, 
                             param1=80, param2=19, minRadius=5, maxRadius=40)

  CIRCLE_DISTANCE_THRESHOLD = 100
  if circles is not None:
    if circles.any() != 0:
      circles = numpy.int16(numpy.around(circles))
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
            return (x,y), img
              
  return (None, None), img

