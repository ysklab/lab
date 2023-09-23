import math
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
class Laser:

  def __init__(self, x, y, theta):
    self.x = x 
    self.y = y
    self.theta = theta # 激光角度
    self.range = 10 # 激光量程10米
    self.res = 1 # 分辨率1度

  def sense(self, obstacles):
    observations = []
    for angle in range(360):
      obs_dist = self.distance_to_segment_at_angle(obstacles, angle)
      if obs_dist is not None:
        observations.append((obs_dist, angle))
      else:
        observations.append((self.range, angle)) 
    return observations

  def distance_to_segment_at_angle(self, obstacles, angle):

    # 激光端点坐标
    lx1 = self.x
    ly1 = self.y
    theta = math.radians(angle) # 转换到弧度
    lx2 = lx1 + self.range * math.cos(theta)
    ly2 = ly1 + self.range * math.sin(theta)

    min_dist = None

    for obs in obstacles:
        x1, y1, x2, y2 = obs.x1, obs.y1, obs.x2, obs.y2
        
        # 与之前函数逻辑相同
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
          dist = math.hypot(lx1 - x1, ly1 - y1)
        elif dx == 0:
            t = (lx1 - x1) / dy
            dist = self.point_dist(t, lx1, ly1, lx2, ly2, x1, y1)
        elif dy == 0:
            t = (ly1 - y1) / dx
            dist = self.point_dist(t, lx1, ly1, lx2, ly2, x1, y1)
        else:
            delta = lx2 - lx1
            gamma = ly2 - ly1
            t = (dx * (ly1 - y1) + dy * (x1 - lx1)) / (delta * dy - delta * dx)
            dist = self.point_dist(t, lx1, ly1, lx2, ly2, x1, y1)

        if min_dist is None or dist < min_dist:
            min_dist = dist

    return min_dist

def point_dist(self, t, lx1, ly1, lx2, ly2, x1, y1):
  px = lx1 + t * (lx2 - lx1)
  py = ly1 + t * (ly2 - ly1)
  return math.hypot(px - lx1, py - ly1)

if __name__ == "__main__":

  # 障碍物...
  obstacles = [
    Obstacle(1, 1, 3, 3),
    Obstacle(2, 4, 4, 2) 
  ]
  # 激光
  laser = Laser(0, 0, 0)

  # 打印观测
  print(laser.sense(obstacles))