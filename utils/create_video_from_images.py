# import cv2
# import numpy as np
# import glob
 
# img_array = []
# for filename in glob.glob('/Users/aditya/Desktop/FYP/project/data/test_temp/test_temp/*.png'):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width,height)
#     img_array.append(img)
 
 
# out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'XVID'), 15, (224, 224))
 
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()



import cv2
import os

image_folder = '/Users/aditya/Desktop/FYP/project/data/test_temp/test_temp/'
video_name = 'video.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
print(height, width, layers)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to 
video = cv2.VideoWriter(video_name, fourcc, 20, (width,height))

for image in images:
    print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()