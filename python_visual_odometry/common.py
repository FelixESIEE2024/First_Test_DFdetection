import numpy as np

def getSubPixelValue(image, pixel):
  
  #bilinear interpolation
  dx = pixel[0]-int(pixel[0])
  dy = pixel[1]-int(pixel[1])

  weight_tl = (1.0 - dx) * (1.0 - dy);
  weight_tr = (dx)       * (1.0 - dy);
  weight_bl = (1.0 - dx) * (dy);
  weight_br = (dx)       * (dy);
             
  f_tl = image[int(pixel[1]),int(pixel[0])]
  f_tr = image[int(pixel[1]),int(pixel[0])+1]
  f_bl = image[int(pixel[1])+1,int(pixel[0])]
  f_br = image[int(pixel[1])+1,int(pixel[0])+1]
     
  return (f_tl*weight_tl + f_tr*weight_tr + f_bl*weight_bl + f_br*weight_br)
  
def filterInvDepthWithVar(invDepth, invDepthVar, camera, lvl):

  filteredInvDepth = (-1.0)*np.ones((camera.height[lvl], camera.width[lvl]), dtype = np.float32)

  for y in range(1, camera.height[lvl]-1):
    for x in range(1, camera.width[lvl]-1):
        
      m = invDepth[y,x]
      s = invDepthVar[y,x]
      if m < 0.0:
        continue
        
      new_m = 0.0
      total_s = 0.0
      for yi in range(-1,2):
        for xi in range(-1,2):
          current_m = invDepth[y+yi,x+xi]
          current_s = invDepthVar[y+yi,x+xi]
          if current_m < 0.0:
            continue
          #if abs(m - current_m) > 1.0*np.sqrt(current_s):
          #  continue
          weight = 1.0/current_s
          new_m += current_m*weight
          total_s += weight
 
      filteredInvDepth[y,x] = new_m/total_s
  
  return filteredInvDepth

              

