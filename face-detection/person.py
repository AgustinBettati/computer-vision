import numpy as np

class Person:

  def __init__(self, name, picture):
    self.name = name
    self.picture_url = picture
    self.img = []
    self.video_img = []
    self.encoding = []
    self.is_present = False
