import cv2
import numpy as np
import copy
from liegroups.numpy import SE3

import frameData
import common

class pose_estimator_gauss_newton:
  def __init__(self, camera, show_debug = True, keyframe_camera = None, frame_camera = None):
    self.lastPoseDiff = SE3.identity() 
    self.show_debug = show_debug
    if frame_camera is None:
      frame_camera = camera
    if keyframe_camera is None:
      keyframe_camera = camera
    self.set_cameras(frame_camera, keyframe_camera)

  def set_cameras(self, frame_camera, keyframe_camera = None):
    if keyframe_camera is None:
      keyframe_camera = frame_camera
    self.camera = frame_camera
    self.frame_camera = frame_camera
    self.keyframe_camera = keyframe_camera

  def _get_camera_pair(self):
    frame_camera = getattr(self, "frame_camera", self.camera)
    keyframe_camera = getattr(self, "keyframe_camera", self.camera)
    return frame_camera, keyframe_camera

  def _validate_level_shapes(self, lvl):
    frame_camera, keyframe_camera = self._get_camera_pair()
    if frame_camera.width[lvl] != keyframe_camera.width[lvl] or frame_camera.height[lvl] != keyframe_camera.height[lvl]:
      raise ValueError("Frame and keyframe cameras must share the same pyramid resolution at each level.")
    return frame_camera, keyframe_camera

  def computeError(self, frame, keyframe, lvl):

    frame_camera, keyframe_camera = self._validate_level_shapes(lvl)

    errorImage = np.zeros((keyframe_camera.height[lvl],keyframe_camera.width[lvl]), dtype = np.float32)

    width = keyframe_camera.width[lvl]
    height = keyframe_camera.height[lvl]
    frame_width = frame_camera.width[lvl]
    frame_height = frame_camera.height[lvl]
    fx = frame_camera.fx[lvl]
    fy = frame_camera.fy[lvl]
    cx = frame_camera.cx[lvl]
    cy = frame_camera.cy[lvl]
    fxinv = keyframe_camera.fxinv[lvl]
    fyinv = keyframe_camera.fyinv[lvl]
    cxinv = keyframe_camera.cxinv[lvl]
    cyinv = keyframe_camera.cyinv[lvl]
    
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


            #Si le point est derriere la camera, on ne le prend pas en compte
            if pointFrame[2] <= 0.0:
                continue;

            pixelFrame = np.array([fx*pointFrame[0]/pointFrame[2] + cx, fy*pointFrame[1]/pointFrame[2] + cy])


            #si le point projeté est en dehors de l'image, on ne le prend pas en compte
            if pixelFrame[0] < 1.0 or pixelFrame[0] >= frame_width-1 or pixelFrame[1] < 1.0 or pixelFrame[1] >= frame_height-1:
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

    frame_camera, keyframe_camera = self._validate_level_shapes(lvl)
  
    width = keyframe_camera.width[lvl]
    height = keyframe_camera.height[lvl]
    frame_width = frame_camera.width[lvl]
    frame_height = frame_camera.height[lvl]
    fx = frame_camera.fx[lvl]
    fy = frame_camera.fy[lvl]
    cx = frame_camera.cx[lvl]
    cy = frame_camera.cy[lvl]
    fxinv = keyframe_camera.fxinv[lvl]
    fyinv = keyframe_camera.fyinv[lvl]
    cxinv = keyframe_camera.cxinv[lvl]
    cyinv = keyframe_camera.cyinv[lvl]
    
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

            if pixelFrame[0] < 1.0 or pixelFrame[0] >= frame_width-1 or pixelFrame[1] < 1.0 or pixelFrame[1] >= frame_height-1.0:
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
    propagatedPose = self.lastPoseDiff.dot(frame.pose)
    bestPose = copy.copy(initialPose)
    
    for lvl in range(4,1,-1):

        frame.pose = copy.copy(bestPose)
        [last_error, errorImage] = self.computeError(frame, keyframe, lvl)

        # The propagated pose from the previous frame is only a prior.
        # If it is already worse than the current pose at the coarsest level,
        # keep the safer initialization to avoid ghosting/drift artefacts.
        if lvl == 4:
            frame.pose = copy.copy(propagatedPose)
            [propagated_error, propagatedErrorImage] = self.computeError(frame, keyframe, lvl)
            if propagated_error < last_error:
                bestPose = copy.copy(propagatedPose)
                last_error = propagated_error
                errorImage = propagatedErrorImage
            else:
                frame.pose = copy.copy(bestPose)
        
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

    frame.pose = copy.copy(bestPose)
    self.lastPoseDiff = bestPose.dot(initialPose.inv())

