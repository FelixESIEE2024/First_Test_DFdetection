import cv2
import numpy as np
import copy
from liegroups.numpy import SE3

import frameData
import common

class pose_estimator_gauss_newton:
  def __init__(self, camera, show_debug = True):
    self.camera = camera
    self.lastPoseDiff = SE3.identity() 
    self.show_debug = show_debug

  def computeError(self, frame, keyframe, lvl):

    errorImage = np.zeros((self.camera.height[lvl],self.camera.width[lvl]), dtype = np.float32)

    width = self.camera.width[lvl]
    height = self.camera.height[lvl]
    fx = self.camera.fx[lvl]
    fy = self.camera.fy[lvl]
    cx = self.camera.cx[lvl]
    cy = self.camera.cy[lvl]
    fxinv = self.camera.fxinv[lvl]
    fyinv = self.camera.fyinv[lvl]
    cxinv = self.camera.cxinv[lvl]
    cyinv = self.camera.cyinv[lvl]
    
    relativePose = frame.pose.dot(keyframe.pose.inv())
    
    residual = 0.0
    num = 0;

    for y in range(0, height):
        for x in range(0, width):
        
            invDepth = keyframe.invDepth[lvl][y,x]
            invDepthVar = keyframe.invDepthVar[lvl][y,x]

            if invDepth <= 0.0:
                continue

            poinKeyframe = np.array([fxinv*x + cxinv, fyinv*y + cyinv,1.0])/invDepth;
            pointFrame = relativePose.dot(poinKeyframe);

            if pointFrame[2] <= 0.0:
                continue;

            pixelFrame = np.array([fx*pointFrame[0]/pointFrame[2] + cx, fy*pointFrame[1]/pointFrame[2] + cy])

            if pixelFrame[0] < 1.0 or pixelFrame[0] >= width-1 or pixelFrame[1] < 1.0 or pixelFrame[1] >= height-1:
                continue;

            vkf = keyframe.image[lvl][y,x]
            #vf = frame.image[lvl][int(pixelFrame[1]),int(pixelFrame[0])]
            #d_f_d_uf = frame.imageDerivative[lvl][int(pixelFrame[1]),int(pixelFrame[0])];
            vf  = common.getSubPixelValue(frame.image[lvl], pixelFrame)
            #d_f_d_uf  = common.getSubPixelValue(frame.imageDerivative[lvl], pixelFrame)

            error = (float(vkf)-float(vf))**2#/invDepthVar
            errorImage[y,x] = error

            residual += error
            num+=1

    if num > 0:
        residual = residual/num;
    else:
        residual = 1000000000000000000.0

    return residual, errorImage
    

  def computeHJPose(self, frame, keyframe, lvl):
  
    width = self.camera.width[lvl]
    height = self.camera.height[lvl]
    fx = self.camera.fx[lvl]
    fy = self.camera.fy[lvl]
    cx = self.camera.cx[lvl]
    cy = self.camera.cy[lvl]
    fxinv = self.camera.fxinv[lvl]
    fyinv = self.camera.fyinv[lvl]
    cxinv = self.camera.cxinv[lvl]
    cyinv = self.camera.cyinv[lvl]
    
    relativePose = frame.pose.dot(keyframe.pose.inv())
    
    J_pose = np.zeros(6)
    H_pose = np.zeros((6,6))

    count = 0;
    for y in range(0, height):
        for  x in range(0, width):
        
            invDepth = keyframe.invDepth[lvl][y,x]
            invDepthVar = keyframe.invDepthVar[lvl][y,x]

            if invDepth <= 0.0:
                continue

            poinKeyframe = np.array([fxinv*x + cxinv, fyinv*y + cyinv,1.0])/invDepth;
            pointFrame = relativePose.dot(poinKeyframe)

            if pointFrame[2] <= 0.0:
                continue

            pixelFrame = np.array([fx*pointFrame[0]/pointFrame[2] + cx, fy*pointFrame[1]/pointFrame[2] + cy]);

            if pixelFrame[0] < 1.0 or pixelFrame[0] >= width-1 or pixelFrame[1] < 1.0 or pixelFrame[1] >= height-1.0:
                continue
                
            vkf = keyframe.image[lvl][y,x]
            #vf = frame.image[lvl][int(pixelFrame[1]),int(pixelFrame[0])]
            #d_f_d_uf = frame.imageDerivative[lvl][int(pixelFrame[1]),int(pixelFrame[0])];
            vf  = common.getSubPixelValue(frame.image[lvl], pixelFrame)
            d_f_d_uf  = common.getSubPixelValue(frame.imageDerivative[lvl], pixelFrame)
                        

            frameInvDepth = 1.0/pointFrame[2]

            v0 = d_f_d_uf[0] * fx * frameInvDepth;
            v1 = d_f_d_uf[1] * fy * frameInvDepth;
            v2 = -(v0 * pointFrame[0] + v1 * pointFrame[1]) * frameInvDepth

            d_I_d_tra = np.array([v0, v1, v2])
            d_I_d_rot = np.array([-pointFrame[2]*v1+pointFrame[1]*v2, pointFrame[2]*v0-pointFrame[0]*v2,-pointFrame[1]*v0+pointFrame[0]*v1])

            residual = (float(vf) - float(vkf))#/invDepthVar

            J = np.array([d_I_d_tra[0], d_I_d_tra[1], d_I_d_tra[2], d_I_d_rot[0], d_I_d_rot[1], d_I_d_rot[2]])
            
            J_pose += J*residual
            count+=1
            for i in range(0,6):
                for j in range(0,6):
                    H_pose[i,j] += J[i]*J[j]
                
    if count > 0:
        J_pose = J_pose/count
        H_pose = H_pose/count
    return [J_pose, H_pose]
         
  def optPose(self, frame, keyframe):
  
    maxIterations = np.array([5, 20, 50, 100, 100]);

    initialPose = copy.copy(frame.pose)
    bestPose = self.lastPoseDiff.dot(frame.pose) 
    #bestPose = copy.copy(frame.pose)
    
    for lvl in range(4,1,-1):
        
        [last_error, errorImage] = self.computeError(frame, keyframe, lvl)
                
        print("lvl: ", lvl, " initial error: ", last_error)
        
        it = 0
        while it < maxIterations[lvl]:
        
            it += 1
            #print("iteration: ", it)
            
            [J_pose, H_pose] = self.computeHJPose(frame, keyframe, lvl)

            lamb = 0.0
            n_try = 0
            
            while True:
            
                H_pose_lambda = copy.copy(H_pose)

                for j in range(0,6):
                    H_pose_lambda[j,j] *= 1.0 + lamb;

                [inc_pose, residuals, rank, s] = np.linalg.lstsq(H_pose_lambda, J_pose, rcond=None)

                #frame.pose = bestPose.dot(SE3.exp(inc_pose))
                frame.pose = bestPose.dot(SE3.exp(inc_pose).inv())
                #frame.pose = SE3.exp(inc_pose).dot(bestPose)
                #frame.pose = (SE3.exp(inc_pose).inv()).dot(bestPose)

                [error, errorImage] = self.computeError(frame, keyframe, lvl)
                #errorNorm = errorImage
                if self.show_debug:
                    error_min = np.amin(errorImage)
                    error_max = np.amax(errorImage)
                    if error_max > error_min:
                        errorNorm = (errorImage - error_min)/(error_max - error_min)
                    else:
                        errorNorm = errorImage
                    cv2.namedWindow("error", cv2.WINDOW_NORMAL)
                    cv2.imshow("error", errorNorm)
                    cv2.waitKey(30)
        
                #print("new error: ", error, "lambda: ", lamb)

                if error < last_error:
                
                    #print("update accepted!")
                    #print("new pose: ")
                    #print(frame.pose)
                    
                    bestPose = copy.copy(frame.pose)

                    self.lastPoseDiff = bestPose.dot(initialPose.inv())

                    p = error / last_error

                    if lamb < 0.2:
                        lamb = 0.0
                    else:
                        lamb *= 0.5

                    last_error = copy.copy(error)

                    if p >  0.999:
                        print(" error improvement too small, level converged! it: ", it, " error: ", last_error, " lambda: ", lamb)
                        it = maxIterations[lvl]
                    break
                else:

                    #print("update rejected!")
                 
                    frame.pose = copy.copy(bestPose)

                    n_try+=1

                    if lamb < 0.2:
                        lamb = 0.2
                    else:
                        lamb *= 2.0#*n_try

                    if inc_pose.dot(inc_pose) < 1e-32:
                        print("update too small, level converged! it: ", it, " error: ", last_error, " lambda: ", lamb)
                        it = maxIterations[lvl]
                        break

