import numpy as np
import params

class camera:
  def __init__(self,fx,fy,cx,cy,width,height):
    
    self.fx = []
    self.fy = []
    self.cx = []
    self.cy = []
    
    self.fxinv = []
    self.fyinv = []
    self.cxinv = []
    self.cyinv = []
    
    self.width = []
    self.height = []
    
    xp = float(params.IMAGE_WIDTH)/width
    yp = float(params.IMAGE_HEIGHT)/height

    out_fx = fx*xp;
    out_fy = fy*yp;
    out_cx = cx*xp;
    out_cy = cy*yp;
    
    for lvl in range(0, params.MAX_LEVELS):
    
        scale = 2.0**lvl

        self.width.append(int(params.IMAGE_WIDTH/scale))
        self.height.append(int(params.IMAGE_HEIGHT/scale))

        self.fx.append(out_fx/scale)
        self.fy.append(out_fy/scale)
        self.cx.append(out_cx/scale)
        self.cy.append(out_cy/scale)
        
        K = np.array(([out_fx/scale,0.0,out_cx/scale],[0.0,out_fy/scale,out_cy/scale],[0.0,0.0,1.0]))
        Kinv = np.linalg.inv(K)
    
        self.fxinv.append(Kinv[0,0])
        self.fyinv.append(Kinv[1,1])
        self.cxinv.append(Kinv[0,2])
        self.cyinv.append(Kinv[1,2])
    
    """
    for lvl in range(0, params.MAX_LEVELS):
      print("lvl: ", lvl)
      print("width, height: ", self.width[lvl], self.height[lvl])
      print("fx, fy, cx, cy: ", self.fx[lvl], " ", self.fy[lvl], " ", self.cx[lvl], " ", self.cy[lvl])
      print("fxinv, fyinv, cxinv, cyinv: ", self.fxinv[lvl], " ", self.fyinv[lvl], " ", self.cxinv[lvl], " ", self.cyinv[lvl])       
    """

