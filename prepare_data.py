#!/usr/bin/env python

import cv2, os.path, sys, math, Image

def Distance (p1, p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)


def ScaleRotateTranslate (image, angle, center = None, new_center = None, scale = None, resample = Image.BICUBIC):

  if (scale is None) and (center is None):
    return image.rotate(angle = angle, resample = resample)
  nx, ny = x,y = center
  sx = sy = 1.0

  if new_center:
    (nx, ny) = new_center

  if scale:
    (sx, sy) = (scale, scale)

  cosine = math.cos(angle)
  sine = math.sin(angle)
  a = cosine / sx
  b = sine / sx
  c = x - nx * a - ny * b
  d = -sine / sy
  e = cosine / sy
  f = y - nx * d - ny * e
  return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample = resample)


def CropFace (image, eye_left = (0,0), eye_right = (0,0), offset_pct = (0.2,0.2), dest_sz = (70,70)):
  offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
  offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
  
  eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
  dist = Distance(eye_left, eye_right)
  reference = dest_sz[0] - 2.0 * offset_h

  scale = float(dist) / float(reference)
  image = ScaleRotateTranslate(image, center = eye_left, angle = rotation)

  crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
  crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)

  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
  image = image.resize(dest_sz, Image.ANTIALIAS)
  return image


def getEyesPosition (im_path):
  face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
  eye_cascade  = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
 
  left_eye = (0, 0)
  right_eye = (0, 0)  

  img = cv2.imread(im_path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))

  for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # FACE GRID
    cv2.rectangle(img, (0, y + h / 2), (1200, y + h / 2), (255, 0, 0), 3)
    cv2.rectangle(img, ((x + (w / 2)), 0), (x+(w / 2), 1200), (255, 0, 0), 4)
    cv2.rectangle(img, (0, y + h / 4), (1200, y + h / 4), (255, 0, 0), 3)
    cv2.rectangle(img, (x, 0), (x, 1200), (255, 0, 0), 3)
    cv2.rectangle(img, (x + w, 0), (x + w, 1200), (255, 0, 0), 3)
    #####

    for (ex, ey, ew, eh) in eyes:
      abs_ex = ex + x
      abs_ey = ey + y
      if (abs_ex + ew / 2 < x + w / 2 and abs_ey + eh / 2 < y + h / 2 and abs_ey + eh > y + h / 4):
        cv2.rectangle(img, (abs_ex, abs_ey), (abs_ex + ew, abs_ey + eh), (0, 0, 255), 3) 
        cv2.rectangle(img, (abs_ex + ew / 2, abs_ey + eh / 2), (abs_ex + ew / 2, abs_ey + eh / 2), (0, 0, 255), 5)
        left_eye = (abs_ex + ew / 2, abs_ey + eh / 2)

      if (abs_ex + ew / 2 > x + w / 2 and abs_ey + eh / 2 < y + h / 2 and abs_ey + eh > y + h / 4):
        cv2.rectangle(img, (abs_ex, abs_ey), (abs_ex + ew, abs_ey + eh), (0, 255, 0), 3) 
        cv2.rectangle(img, (abs_ex + ew / 2, abs_ey + eh / 2), (abs_ex + ew / 2, abs_ey + eh / 2), (0, 0, 255), 5)
        right_eye = (abs_ex + ew / 2, abs_ey + eh / 2)

    if (left_eye == (0, 0)):
      left_eye =  (x + w / 4, y + 2 * h / 5)

    if (right_eye == (0, 0)):
      right_eye = (x + 3 * w / 4, y + 2 * h / 5)

  #cv2.imshow('img', img)   
  #cv2.waitKey(0)
  return(left_eye, right_eye)
      

def preparePhoto (path):
  image = Image.open(path)
  dist_path = "photos/" + '/'.join(path.rstrip().split('/')[1:])
  dist_directory = '/'.join(dist_path.rstrip().split('/')[:2])
  eye_left_pos, eye_right_pos = getEyesPosition(path)

  if not os.path.exists(dist_directory):
    os.makedirs(dist_directory)
  CropFace(image, eye_left = eye_left_pos, eye_right = eye_right_pos, offset_pct=(0.3, 0.3), dest_sz=(200, 200)).save(dist_path)
  print ("%s SAVED!\tLeft Eye: %s\t Right Eye: %s" % ('/'.join(path.rstrip().split('/')[1:]), eye_left_pos, eye_right_pos))

if __name__ == "__main__":
  if (len(sys.argv) != 2):
    print("Wrong parameters!")
    sys.exit()

  FOLDER_NAME = sys.argv[1]
  SEPARATOR = ";"
 
  label = 0
  csv_file = open('data.txt', 'w')

  for dirname, dirnames, filenames in os.walk(FOLDER_NAME):

    for subdirname in dirnames:
      subject_path = os.path.join(dirname, subdirname)

      for filename in os.listdir(subject_path):
        abs_path = "%s/%s" % (subject_path, filename)
        csv_file.write("%s%s%d\n" % ("photos/" + '/'.join(abs_path.rstrip().split('/')[1:]), SEPARATOR, label))
        preparePhoto(abs_path)
      label = label + 1
  csv_file.close()
