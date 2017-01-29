import socket
import robot_actions as robot
import time
import os
import numpy as np
import cv2
# Python Image Library
import Image
import thread
import math
import motion
from multiprocessing import Value,Queue
from naoqi import ALProxy
from naoqi import ALBroker
from naoqi import ALModule

from optparse import OptionParser

robotIp = "192.168.1.103"  
port=9559

camProxy = ALProxy("ALVideoDevice", robotIp, port)
resolution = 1    # VGA
colorSpace = 0   # Y channel

upper_cam = camProxy.subscribeCamera("Ucam",0, resolution, colorSpace, 5)
depth = camProxy.subscribeCamera("Dcam",2, resolution, colorSpace, 5)

basic_awareness = ALProxy("ALBasicAwareness",robotIp, port)
motionProxy  = ALProxy("ALMotion", robotIp, port)
basic_awareness.setStimulusDetectionEnabled("People",True)
basic_awareness.setStimulusDetectionEnabled("Movement",True)
basic_awareness.setStimulusDetectionEnabled("Sound",True)
basic_awareness.setStimulusDetectionEnabled("Touch",True)

basic_awareness.setParameter("LookStimulusSpeed",0.7)
basic_awareness.setParameter("LookBackSpeed",0.5)
basic_awareness.setEngagementMode("FullyEngaged")
basic_awareness.setTrackingMode("Head")

tracker = ALProxy("ALTracker", robotIp, port)
targetName = "Face"
faceWidth = 0.1
tracker.registerTarget(targetName, faceWidth)


def check_head_limit(step,num):
	basic_awareness.startAwareness()
	tracker.track(targetName)
	while num.value==0:
		name  = "Head"
		frame = motion.FRAME_TORSO 
		useSensorValues = True
		result  = motionProxy.getPosition(name, frame, useSensorValues)
		if float(result[5])<-0.3 or float(result[5])>0.3:
			basic_awareness.stopAwareness()
			tracker.stopTracker()
			break  
	basic_awareness.stopAwareness()
	tracker.stopTracker()

def cam(step,num):
	"""
	First get an image from Nao, then show it on the screen with PIL.
	"""
	myBroker = ALBroker("myBroker","0.0.0.0", 0, robotIp, 9559)   

	global HumanGreeter
	HumanGreeter = HumanGreeterModule("HumanGreeter")

	e=open('files/episode.txt','rb')
	ep=e.read()
	e.close()
   
	#save_path1='dataset/RGB/ep'+str(ep[2])+'/'
	#save_path2='dataset/Depth/ep'+str(ep[2])+'/'
	save_path1='dataset/RGB/ep'+str(ep[2])+str(ep[3])+'/'
	save_path2='dataset/Depth/ep'+str(ep[2])+str(ep[3])+'/'
	t0 = time.time()
	
	# Get a camera image.
	# image[6] contains the image data passed as an array of ASCII chars.
	for i in range(1,9):
		yimg = camProxy.getImageRemote(upper_cam)
		dimg = camProxy.getImageRemote(depth)
		image=np.zeros((dimg[1], dimg[0]),np.uint8)
		values=map(ord,list(dimg[6]))
		j=0
		for y in range (0,dimg[1]):
			for x in range (0,dimg[0]):
				image.itemset((y,x),values[j])
				j=j+1
		name="depth_"+str(step)+"_"+str(i)+".png"
		complete_depth=os.path.join(save_path2,name)
		cv2.imwrite(complete_depth,image)
		im = Image.fromstring("L", (yimg[0], yimg[1]), yimg[6])
		name="image_"+str(step)+"_"+str(i)+".png"
		complete_rgb=os.path.join(save_path1,name)
		im.save(complete_rgb, "PNG")	


	myBroker.shutdown()   
	t1 = time.time()

	# Time the image transfer.

	print "acquisition delay ", t1 - t0
	num.value=1
	

def hi():

	global flag
	flag=1

class HumanGreeterModule(ALModule):
   
	def __init__(self, name):
		ALModule.__init__(self, name)
		#self.tts = ALProxy("ALTextToSpeech")
		global memory
		memory = ALProxy("ALMemory")
		memory.subscribeToEvent("FaceDetected","HumanGreeter","onFaceDetected")
		#memory.subscribeToEvent("PeoplePerception/PeopleDetected","HumanGreeter","onFaceDetected")
	def onFaceDetected(self, *_args):

		memory.unsubscribeToEvent("FaceDetected","HumanGreeter")
		#memory.unsubscribeToEvent("PeoplePerception/PeopleDetected","HumanGreeter")
		hi()
		#PeoplePerception/PeopleDetected
		#memory.subscribeToEvent("FaceDetected","HumanGreeter","onFaceDetected")

l_curr=0
l_prev=0
def location(x):
	if 90<=x<110:
		return 0
	elif 60<=x<90:
		return -1
	elif 0<x<60:
		return -1
	elif 110<x<=140:
		return 1
	elif 140<x<=198:
		return 1


def rotate(l_curr,l_prev):
	a=l_curr+l_prev
	if a<-1 or a>1:
		a=0
	sign=a-l_prev
	theta=1*sign * math.pi/9
	motionProxy.moveTo(0, 0, theta)
	return a

def abs_rot(l_prev):
	
	theta=-1 *l_prev* math.pi/9
	motionProxy.moveTo(0, 0, theta)
	return 0

host='192.168.1.102'
port=12375
s2=socket.socket()
s2.bind((host,port))
s2.listen(5)                 # Now wait for client connection.
step=1

while step<=1010:
	c, addr = s2.accept() 
	d = c.recv(1024)
	global flag
	flag=0
	data= d.split(':', 2 )
	num2= Value('d', 0.0)
	num3= Value('d', 0.0)
	r=str(0)
	if float(data[1]) == -1:
		l_prev=abs_rot(l_prev)
	else:
		l_curr=location(float(data[1]))
		l_prev=rotate(l_curr,l_prev)
	
	
	if str(str(data[0]))=='1' or str(str(data[0]))=='-' :
		thread.start_new_thread(cam,(step,num2,))
		r=robot.main(data[0],step)
	elif str(str(data[0]))=='2':
		thread.start_new_thread(cam,(step,num2,))
		thread.start_new_thread(check_head_limit,(step,num3,))
		r=robot.main(data[0],step)
	elif str(str(data[0]))=='3':
		thread.start_new_thread(cam,(step,num2,))
		thread.start_new_thread(check_head_limit,(step,num3,))
		r=robot.main(data[0],step)
	elif str(str(data[0]))=='4':
		thread.start_new_thread(cam,(step,num2,))
		thread.start_new_thread(check_head_limit,(step,num3,))
		r=robot.main(data[0],step)		
	while num2.value==0:
		pass
	num3.value=1
	names =['HeadYaw','HeadPitch']
	motionProxy.setAngles(names,[0.0,-0.26179],0.2)
	s=str(r)+' '+str(flag)+'\n'	
	if flag==1:
		print "human detected"
		flag=0
	print s
	c.send(s)
	step=step+1

camProxy.unsubscribe(upper_cam)
camProxy.unsubscribe(depth)	

		
