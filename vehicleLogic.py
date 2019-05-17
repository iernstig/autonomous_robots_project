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
from util import *
################################################################################
# This dictionary contains all distance values to be filled by function onDistance(...).
distances = { "front": 0.0, "left": 0.0, "right": 0.0, "rear": 0.0 };

################################################################################
def onDistance(msg, senderStamp, timeStamps):
    #print "Received distance; senderStamp=" + str(senderStamp)
    #print "sent: " + str(timeStamps[0]) + ", received: " + str(timeStamps[1]) + ", sample time stamps: " + str(timeStamps[2])
    #print msg
    if senderStamp == 0:
        distances["front"] = msg.distance
    if senderStamp == 1:
        distances["left"] = msg.distance
    if senderStamp == 2:
        distances["rear"] = msg.distance
    if senderStamp == 3:
        distances["right"] = msg.distance
# This callback is triggered whenever there is a new distance reading coming in.

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
# Load calibration data
#xGrid = numpy.loadtxt("xGrid.csv", delimiter=",")
#yGrid = numpy.loadtxt("yGrid.csv", delimiter=",")

# integral part init
integralPart = 0
aimPoint = (0,0)
# Counter to eval fram rate
if (cid == 112):
  frameCounter = 0
  counterTime = time.time()
# Main loop to process the next image frame coming in.
#i=0

while True:
    # Wait for next notification.
    cond.Z()
    #print "Received new frame."
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
    img = img[220:330,:,:]

    ############################################################################
    # TODO: Add some image processing logic here.

    # The following example is adding a red rectangle and displaying the result.
    # cv2.rectangle(img, (50, 50), (100, 100), (0,0,255), 2)

    # Added by Erik
    # canny = cv2.Canny(img, 100,200)
    #if (i % 40 == 0):
    #  cv2.imwrite("screen-" + str(i) +".png", img)
    #  print("Grabbed screen nr " + str(i))
    #i = i +1

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #canny = cv2.Canny(gray_img, 110, 220)
    gray_img = cv2.medianBlur(gray_img, 5)
    
    # image colors 
    hsv_low_blue = (100, 85, 45)
    hsv_high_blue = (120, 255, 90)
    hsv_low_yellow = (25, 80, 120)
    hsv_high_yellow = (32, 255, 255)
    hsv_low_orange = cv2.cvtColor((156, 91, 83), cv2.COLOR_RGB2HSV)
    hsv_high_orange = cv2.cvtColor((209, 156, 151), cv2.COLOR_RGB2HSV)
    
    blue_cones = cv2.inRange(hsv_img, hsv_low_blue, hsv_high_blue)
    yellow_cones = cv2.inRange(hsv_img, hsv_low_yellow, hsv_high_yellow)
    orange_cones = cv2.inRange(hsv_img, hsv_low_orange, hsv_high_orange)
    #result = cv2.bitwise_and(img, img, mask=mask)

    # Dilate 
    kernel = numpy.ones((3,3), numpy.uint8)
    dilate_blue = cv2.dilate(blue_cones, kernel, iterations=4)
    dilate_yellow = cv2.dilate(yellow_cones, kernel, iterations=4)
    dilate_orange = cv2.dilate(orange_cones, kernel, iterations=4)

    # Erode
    erode_blue = cv2.erode(dilate_blue, kernel, iterations=2)
    erode_yellow = cv2.erode(dilate_yellow, kernel, iterations=2)
    erode_orange = cv2.erode(dilate_orange, kernel, iterations=2)


    blue_list, yellow_list, img = findCones(erode_blue, erode_yellow, img, cid)
    orange_list, img = detectOrangeCones(erode_orange, img, cid)
    # img, circle_data = findCircles(gray_img, img)
    # yellow_list, img = filterHitsOnCar(yellow_list, circle_data, distance_thres=80, image=img)
    # blue_list, img = filterHitsOnCar(blue_list, circle_data, distance_thres=80, image=img)

    aimPoint = calcAimPoint(blue_list, yellow_list, aimPoint)
    if (aimPoint[0] is not None):
      img = cv2.drawMarker(img, position=aimPoint, color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=3)
      
      (steeringAngle, integralPart) = calcSteeringAngle(aimPoint, integralPart)
      #img = cv2.putText(img, text=str(steeringAngle/numpy.pi*180), org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 255, 255), lineType = 2)
    else:
      img = cv2.drawMarker(img, position=(0,0), color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=3)
   
    img = cv2.putText(img, text=str(distances["front"]), org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255,255,255), lineType = 2)
    
    car_detect_img = detectCarCircles(img.copy())
    
    if(cid == 253):
      #cv2.imshow("image", img)
      cv2.imshow('car-detect', car_detect_img)
      #cv2.imshow('canny', canny)
      #cv2.imshow("Gray", gray_img)
      #cv2.imshow("canny", canny)
      #cv2.imshow("mask", mask)
      #cv2.imshow("result", result)
      #cv2.imshow("Blue original", blue_cones)
      #cv2.imshow("Yellow original", yellow_cones)
      #cv2.imshow("Blue Eroded", erode_blue)
      #cv2.imshow("Yellow Eroded", erode_yellow)
      #cv2.imshow("Cones", cone_image) 
      cv2.waitKey(2)

    ############################################################################
    # Example: Accessing the distance readings.
    '''print "Front = " + str(distances["front"])
    print "Left = " + str(distances["left"])
    print "Right = " + str(distances["right"])
    print "Rear = " + str(distances["rear"])'''

    ############################################################################
    # Example for creating and sending a message to other microservices; can
    # be removed when not needed.
    '''angleReading = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_AngleReading()
    angleReading.angle = 123.45

    # 1038 is the message ID for opendlv.proxy.AngleReading
    session.send(1038, angleReading.SerializeToString());'''

    ############################################################################
    # Steering and acceleration/decelration.
    #
    # Uncomment the following lines to steer; range: +38deg (left) .. -38deg (right).
    # Value groundSteeringRequest.groundSteering must be given in radians (DEG/180. * PI).
    #print(str(distances["front"]))
    if (aimPoint[0] is not None):
      #print("Steering angle: " + str(steeringAngle/numpy.pi*180))
      groundSteeringRequest = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_GroundSteeringRequest()
      groundSteeringRequest.groundSteering = steeringAngle
      session.send(1090, groundSteeringRequest.SerializeToString());

    # Uncomment the following lines to accelerate/decelerate; range: +0.25 (forward) .. -1.0 (backwards).
    # Be careful!
    pedalPositionRequest = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_PedalPositionRequest()
    if (distances["front"] > 0.4):
      pedalPositionRequest.position = 0.11
    elif (distances["front"] > 0.35):
      print("Front distance close!")
      pedalPositionRequest.position = 0.08
    else:
      print("Front distance too close!")
      pedalPositionRequest.position = 0
      groundSteeringRequest = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_GroundSteeringRequest()
      groundSteeringRequest.groundSteering = 0
      session.send(1090, groundSteeringRequest.SerializeToString());
      
    session.send(1086, pedalPositionRequest.SerializeToString());

