import cv2
import copy
import numpy as np
from liegroups.numpy import SE3

import params

class frameData:
  def __init__(self):
    self.image = []
    self.imageDerivative = []  
    self.invDepth = []
    self.invDepthVar = []
    self.pose = SE3.identity()  
            
    for lvl in range(0, params.MAX_LEVELS):
      width = int(params.IMAGE_WIDTH/(2**lvl))
      height = int(params.IMAGE_HEIGHT/(2**lvl))
      self.image.append(np.zeros((height,width)))
      self.imageDerivative.append(np.zeros((height,width,2),dtype = np.float32)) 
      self.invDepth.append(np.zeros((height,width),dtype = np.float32))
      self.invDepthVar.append(np.zeros((height,width),dtype = np.float32))
            
  def computeDerivative(self, lvl):
    #self.imageDerivative[lvl][:,1:-1,0] = (self.image[lvl][:,1:-1] - self.image[lvl][:,0:-2])/1.0
    #self.imageDerivative[lvl][1:-1,:,1] = (self.image[lvl][1:-1,:] - self.image[lvl][0:-2,:])/1.0
    #self.imageDerivative[lvl][:,1:-2,0] = (self.image[lvl][:,2:-1] - self.image[lvl][:,0:-3])/2.0
    #self.imageDerivative[lvl][1:-2,:,1] = (self.image[lvl][2:-1,:] - self.image[lvl][0:-3,:])/2.0
    #self.imageDerivative[lvl][:,2:-3,0] = -(1.0/12.0)*self.image[lvl][:,4:-1] + (8.0/12.0)*self.image[lvl][:,3:-2] - (8.0/12.0)*self.image[lvl][:,1:-4] + (1.0/12.0)*self.image[lvl][:,0:-5]
    #self.imageDerivative[lvl][2:-3,:,1] = -(1.0/12.0)*self.image[lvl][4:-1,:] + (8.0/12.0)*self.image[lvl][3:-2,:] - (8.0/12.0)*self.image[lvl][1:-4,:] + (1.0/12.0)*self.image[lvl][0:-5,:]
    
    #self.imageDerivative[lvl][:,:,0] = cv2.Sobel(self.image[lvl],cv2.CV_32F,1,0,ksize=5)
    #self.imageDerivative[lvl][:,:,1] = cv2.Sobel(self.image[lvl],cv2.CV_32F,0,1,ksize=5)

    for y in range(1, self.image[lvl].shape[0]-1):
      for x in range(1, self.image[lvl].shape[1]-1):
        self.imageDerivative[lvl][y,x,0] = (float(self.image[lvl][y,x+1]) - float(self.image[lvl][y,x-1]))/2.0
        self.imageDerivative[lvl][y,x,1] = (float(self.image[lvl][y+1,x]) - float(self.image[lvl][y-1,x]))/2.0

  def setImage(self, image):
    for lvl in range(0, params.MAX_LEVELS):
      width = int(params.IMAGE_WIDTH/(2**lvl))
      height = int(params.IMAGE_HEIGHT/(2**lvl))
      self.image[lvl] = cv2.resize(image, (width, height), cv2.INTER_AREA)
      self.computeDerivative(lvl)
      """
      if lvl == 0:
        self.computeDerivative(lvl)
      else:
        self.imageDerivative[lvl] = cv2.resize(self.imageDerivative[0], (width, height), cv2.INTER_AREA)        
      """      
  def setImageAndPose(self, image, pose):
    self.setImage(image)
    self.pose = copy.copy(pose)

  def setInvDepth(self, _invDepth, _invDepthVar):
    for lvl in range(0, params.MAX_LEVELS):
      width = int(params.IMAGE_WIDTH/(2**lvl))
      height = int(params.IMAGE_HEIGHT/(2**lvl))
      self.invDepth[lvl] = cv2.resize(_invDepth, (width, height), cv2.INTER_AREA)
      self.invDepthVar[lvl] = cv2.resize(_invDepthVar, (width, height), cv2.INTER_AREA)
