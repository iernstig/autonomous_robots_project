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

def calcAimPoint(blueCones, yellowCones):
    (width, height) = blueCones.shape
    kernel = numpy.ones((10,10))/(10.0*10.0)
    anchor = (-1,-1)
    delta = 0
    ddepth = -1
    blueCoords = None
    yellowCoords = None
    nonZeroBlue = cv2.findNonZero(blueCones)
    nonZeroYellow = cv2.findNonZero(yellowCones)
    blueCoords = (0,481)
    yellowCoords = (0,481)
    if (nonZeroBlue is not None):
        for i in range(nonZeroBlue.shape[0]):
            x = nonZeroBlue[i,0,0]
            y = nonZeroBlue[i,0,1]
            if (y < blueCoords[1]):         
                blueCoords = (x,y)
    else:
        blueCoords = None
    if (nonZeroYellow is not None):
        for i in range(nonZeroYellow.shape[0]):
            x = nonZeroYellow[i,0,0]
            y = nonZeroYellow[i,0,1]
            if (y < yellowCoords[1]):         
                yellowCoords = (x,y)
    else:
        yellowCoords = None
    '''
    for j in range(height/3, height):
        for i in range(width):
            if (blueCones[i,j] == 1):
                blueCoords = (i, j)
                break
    for j in range(height/3, height):
        for i in range(width):
            if (yellowCones[i,j] == 1):
                yellowCoords = (i, j)
                break
    '''
    if (blueCoords is not None and yellowCoords is not None):
        xCoord = (yellowCoords[0] + blueCoords[0])/2
        yCoord = (yellowCoords[1] + blueCoords[1])/2
    else:
        xCoord = None
        yCoord = None
    return (xCoord, yCoord)


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
# Main loop to process the next image frame coming in.
while True:
    # Wait for next notification.
    cond.Z()
    print "Received new frame."

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

    ############################################################################
    # TODO: Add some image processing logic here.

    # The following example is adding a red rectangle and displaying the result.
    # cv2.rectangle(img, (50, 50), (100, 100), (0,0,255), 2)

    # Added by Erik
    #canny = cv2.Canny(img, 100,200)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_low_blue = (105, 80, 40) # for blue
    hsv_high_blue = (125, 255, 255) # for blue
    hsv_low_yellow = (25, 80, 40) # for yellow
    hsv_high_yellow = (35, 255, 255) # for yellow
    blue_cones = cv2.inRange(hsv_img, hsv_low_blue, hsv_high_blue)
    yellow_cones = cv2.inRange(hsv_img, hsv_low_yellow, hsv_high_yellow)
    #result = cv2.bitwise_and(img, img, mask=mask)

    # Dilate 
    kernel = numpy.ones((3,3), numpy.uint8)
    dilate_blue = cv2.dilate(blue_cones, kernel, iterations=3)
    dilate_yellow = cv2.dilate(yellow_cones, kernel, iterations=3)

    # Erode
    erode_blue = cv2.erode(dilate_blue, kernel, iterations=2)
    erode_yellow = cv2.erode(dilate_yellow, kernel, iterations=2)

    cone_image = numpy.zeros((480,640,3), numpy.uint8)
    blue_image = numpy.zeros((480,640,3), numpy.uint8)
    blue_image[:] = (255,0,0)
    yellow_image = numpy.zeros((480,640,3), numpy.uint8)
    yellow_image[:] = (0, 255, 255)

    yellow_part = cv2.bitwise_and(yellow_image, yellow_image, mask=erode_yellow)
    blue_part = cv2.bitwise_and(blue_image, blue_image, mask=erode_blue)
    cone_image = cv2.add(blue_part, yellow_part)

    aimPoint = calcAimPoint(erode_blue, erode_yellow)
    if (aimPoint[0] is not None):
      img = cv2.drawMarker(img, position=aimPoint, color=(0,0,255), markerType=cv2.MARKER_CROSS)
    else:
      img = cv2.drawMarker(img, position=(0,0), color=(0,0,255), markerType=cv2.MARKER_CROSS)
   

    if(cid == 253):
      cv2.imshow("image", img);
      #cv2.imshow("canny", canny)
      #cv2.imshow("mask", mask)
      #cv2.imshow("result", result)
      #cv2.imshow("Dilated", dilate)
      #cv2.imshow("Blue", erode_blue)
      #cv2.imshow("Yellow", erode_yellow)
      cv2.imshow("Cones", cone_image)
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
    angleReading = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_AngleReading()
    angleReading.angle = 123.45

    # 1038 is the message ID for opendlv.proxy.AngleReading
    session.send(1038, angleReading.SerializeToString());

    ############################################################################
    # Steering and acceleration/decelration.
    #
    # Uncomment the following lines to steer; range: +38deg (left) .. -38deg (right).
    # Value groundSteeringRequest.groundSteering must be given in radians (DEG/180. * PI).
    #groundSteeringRequest = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_GroundSteeringRequest()
    #groundSteeringRequest.groundSteering = 0
    #session.send(1090, groundSteeringRequest.SerializeToString());

    # Uncomment the following lines to accelerate/decelerate; range: +0.25 (forward) .. -1.0 (backwards).
    # Be careful!
    #pedalPositionRequest = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_PedalPositionRequest()
    #pedalPositionRequest.position = 0
    #session.send(1086, pedalPositionRequest.SerializeToString());

