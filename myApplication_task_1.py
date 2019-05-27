#!/usr/bin/env python2

# Copyright (C) 2018 Christian Berger
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# sysv_ipc is needed to access the shared memory where the camera image is present.
import sysv_ipc
# numpy and cv2 are needed to access, modify, or display the pixels
import numpy
import cv2
import time
# OD4Session is needed to send and receive messages
import OD4Session
# Import the OpenDLV Standard Message Set.
import opendlv_standard_message_set_v0_9_6_pb2

################################################################################
# This dictionary contains all distance values to be filled by function onDistance(...).
distances = { "front": 0.0, "left": 0.0, "right": 0.0, "rear": 0.0 };

################################################################################
# This callback is triggered whenever there is a new distance reading coming in.
def onDistance(msg, senderStamp, timeStamps):
    if senderStamp == 0:
        distances["front"] = msg.distance
    if senderStamp == 1:
        distances["left"] = msg.distance
    if senderStamp == 2:
        distances["rear"] = msg.distance
    if senderStamp == 3:
        distances["right"] = msg.distance


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
  
def calcSteeringAngle(aimPoint, integralPart):
    '''
    Calculating steering angle based on aimpoint, implemented as a PI-regulator,
    however the integral part has been put to zero. The regulator tries to keep the
    x-coordinate of the aimpoint in the middle of the image.
    '''
    # Different gains for left and right steering since the car has a bias.
    K_p_left = 0.28
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
    # Keep the steering angle within allowed bounds.
    if (steeringAngle < -0.3):
        steeringAngle = -0.3
        integralPart = 0
    elif (steeringAngle > 0.3):
        steeringAngle = 0.3
        integralPart = 0
    return steeringAngle, integralPart

def findCones(blue_img, yellow_img, image, cid):
  '''
  Find cones from blue and yellow color filtered binary images, using openCVs findContours. Hits are 
  filtered out if they are too small or large to avoid some false positives. The contours are 
  plotted if in replay mode.
  '''
  im2, cnts, hier = cv2.findContours(blue_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
  
  im2, cnts, hier = cv2.findContours(yellow_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  yellow_hits = []
  for c in cnts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    area = cv2.contourArea(c)
    if (area < 2000 and area > 50):
      if (cid == 253):
        cv2.drawContours(image, [c], 0, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)
        cv2.putText(image, "center", (cX-10, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
      yellow_hits.append((cX, cY))
  return blue_hits, yellow_hits, image


# Create a session to send and receive messages from a running OD4Session;
# Replay mode: CID = 253
# Live mode: CID = 112
# TODO: Change to CID 112 when this program is used on Kiwi.
cid = 253
session = OD4Session.OD4Session(cid)
# Register a handler for a message; the following example is listening
# for messageID 1039 which represents opendlv.proxy.DistanceReading.
# Cf. here: https://github.com/chalmers-revere/opendlv.standard-message-set/blob/master/opendlv.odvd#L113-L115
messageIDDistanceReading = 1039
session.registerMessageCallback(messageIDDistanceReading, onDistance, opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_DistanceReading)
# Connect to the network session.
session.connect()

################################################################################
# The following lines connect to the camera frame that resides in shared memory.
# This name must match with the name used in the h264-decoder-viewer.yml file.
name = "/tmp/img.argb"
# Obtain the keys for the shared memory and semaphores.
keySharedMemory = sysv_ipc.ftok(name, 1, True)
keySemMutex = sysv_ipc.ftok(name, 2, True)
keySemCondition = sysv_ipc.ftok(name, 3, True)
# Instantiate the SharedMemory and Semaphore objects.
shm = sysv_ipc.SharedMemory(keySharedMemory)
mutex = sysv_ipc.Semaphore(keySemCondition)
cond = sysv_ipc.Semaphore(keySemCondition)

################################################################################
# integral part init
integralPart = 0
aimPoint = (0,0)
# Counter to eval fram rate
if (cid == 112):
  frameCounter = 0
  counterTime = time.time()
# Main loop to process the next image frame coming in.

while True:
    # Wait for next notification.
    cond.Z()
    # Loop that prints frame rate when live on Kiwi
    if (cid == 112):
      frameCounter += 1
      if frameCounter == 20:
        timeElapsed = time.time() - counterTime
        frameRate = 20/timeElapsed
        print("************")
        print("Frame rate is: " + str(frameRate) + "per second!")
        print("************")
        frameCounter = 0
        counterTime = time.time()
      
    # Lock access to shared memory.
    mutex.acquire()
    # Attach to shared memory.
    shm.attach()
    # Read shared memory into own buffer.
    buf = shm.read()
    # Detach to shared memory.
    shm.detach()
    # Unlock access to shared memory.
    mutex.release()

    # Turn buf into img array (640 * 480 * 4 bytes (ARGB)) to be used with OpenCV.
    img = numpy.frombuffer(buf, numpy.uint8).reshape(480, 640, 4)
    # Crop unneccesary parts of the image
    img = img[220:330,:,:]

    ############################################################################

    # Convert BGR image to HSV 
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Filter colors in HSV space 
    hsv_low_blue = (100, 85, 45)
    hsv_high_blue = (120, 255, 90)
    hsv_low_yellow = (25, 80, 120)
    hsv_high_yellow = (32, 255, 255)
    blue_cones = cv2.inRange(hsv_img, hsv_low_blue, hsv_high_blue)
    yellow_cones = cv2.inRange(hsv_img, hsv_low_yellow, hsv_high_yellow)

    # Dilate 
    kernel = numpy.ones((3,3), numpy.uint8)
    dilate_blue = cv2.dilate(blue_cones, kernel, iterations=4)
    dilate_yellow = cv2.dilate(yellow_cones, kernel, iterations=4)

    # Erode
    erode_blue = cv2.erode(dilate_blue, kernel, iterations=2)
    erode_yellow = cv2.erode(dilate_yellow, kernel, iterations=2)

    # Find cones of different colors
    blue_list, yellow_list, img = findCones(erode_blue, erode_yellow, img, cid)

    aimPoint = calcAimPoint(blue_list, yellow_list, aimPoint)

    # Draw aimpoint to be used when in replay mode
    img = cv2.drawMarker(img, position=aimPoint, color=(0,0,255), markerType=cv2.MARKER_CROSS)

    # Calculate steering angle based on aimpoint
    (steeringAngle, integralPart) = calcSteeringAngle(aimPoint, integralPart)

    # Display image if in replay mode
    if(cid == 253):
      cv2.imshow("image", img)
      cv2.waitKey(2)

    ############################################################################
    # Steering and acceleration/decelration.
    #
    # Uncomment the following lines to steer; range: +38deg (left) .. -38deg (right).
    # Value groundSteeringRequest.groundSteering must be given in radians (DEG/180. * PI).

    groundSteeringRequest = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_GroundSteeringRequest()
    groundSteeringRequest.groundSteering = steeringAngle
    

    # Uncomment the following lines to accelerate/decelerate; range: +0.25 (forward) .. -1.0 (backwards).
    # Be careful!
    pedalPositionRequest = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_PedalPositionRequest()
    if (distances["front"] > 0.4):
      pedalPositionRequest.position = 0.14
    elif (distances["front"] > 0.35):
      print("Front distance close!")
      pedalPositionRequest.position = 0.08
    else:
      print("Front distance too close!")
      pedalPositionRequest.position = 0
      groundSteeringRequest.groundSteering = 0
    
    session.send(1090, groundSteeringRequest.SerializeToString());
    session.send(1086, pedalPositionRequest.SerializeToString());

