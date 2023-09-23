import math

# 障碍物类
class Obstacle:
  def __init__(self, x1, y1, x2, y2):
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2 
    self.y2 = y2

def distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx**2 + dy**2)
# 激光雷达类  
class Laser:
  def __init__(self, x, y, theta):
    self.x = x
    self.y = y 
    self.theta = theta

  def sense(self, obstacles):
    observations = []
    for obs in obstacles:
      dist = self.distance_to_segment(obs)
      if dist is not None:
        observations.append(dist)
    return observations
  

  def distance_to_segment(self, obs):

    x1, y1 = obs.x1, obs.y1
    x2, y2 = obs.x2, obs.y2

    lx1, ly1 = self.x, self.y
    theta = self.theta
    lx2 = self.x + math.cos(theta) 
    ly2 = self.y + math.sin(theta)

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return None
    
    elif dx == 0:
        t = (lx1 - x1) / dy

    elif dy == 0:
        t = (ly1 - y1) / dx

    else:
        delta = lx2 - lx1
        gamma = ly2 - ly1

        if delta == 0 and gamma == 0:
          return math.sqrt((lx1 - x1)**2 + (ly1 - y1)**2)
    
        # 新增判断平行重合的情况
        if delta * dy - delta * dx == 0:
            return min(distance(lx1, ly1, x1, y1), distance(lx1, ly1, x2, y2))

        t = (dx * (ly1 - y1) + dy * (x1 - lx1)) / (delta * dy - delta * dx)

    # 判断t是否在0-1区间内
    if 0 < t < 1:  
        px = lx1 + delta * t
        py = ly1 + gamma * t
        return math.sqrt((px - self.x)**2 + (py - self.y)**2)

    else:
        return None

if __name__ == "__main__":
  
  # 障碍物列表
  obstacles = [
    Obstacle(1, 1, 3, 3),
    Obstacle(2, 4, 4, 2) 
  ]

  # 激光雷达 
  laser = Laser(0, 0, math.pi / 6) 

  # 打印观测结果
  print(laser.sense(obstacles))