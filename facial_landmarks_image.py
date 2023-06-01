import cv2 
import numpy as np
import dlib
import argparse

# define the CLI parameters
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help='The image with a face from where you want to detect the landmarks', required=False, default='C:/Users/pcteste1/Downloads/profile_image.jpg')
parser.add_argument('-l', '--landmarks', help='The path to the landmarks file', required=False, default="C:\\Users\\pcteste1\\Documents\\Gitkraken\\CV_Landmarks\\shape_predictor_68_face_landmarks.dat")

# convert the arguments
args = vars(parser.parse_args())

# load the image
img = cv2.imread(args['image'])
assert img is not None, 'Image not found'

# reduce the image size
img = cv2.resize(img, None, fx=0.2, fy=0.2)

# convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# face detector
face_detector = dlib.get_frontal_face_detector()

# landmark points
landmarks = dlib.shape_predictor(args['landmarks'])
assert landmarks is not None, "Landmarks file not found. Please change the path"

# find coordinates of the face
# The 1 in the second argument indicates that we should upsample the image 1 time
rects = face_detector(gray_img, 1)

# for each deteted face
for (i, rect) in enumerate(rects):
    # find the landmarks in the image
    reference_point = landmarks(gray_img, rect)

    # convert the landmark points to (x,y) coordinates
    coords = np.zeros((reference_point.num_parts, 2), dtype=np.int32)
    for i in range(0, reference_point.num_parts):
        coords[i] = (reference_point.part(i).x, reference_point.part(i).y)

    # draw the landmark points int the image
    for (x, y) in coords:
        cv2.circle(img, (x,y), 2, (255, 0, 0), 2)

    # unpack the coordinates and transform them to a bounding box
    x, y, w, h = rect.left(), rect.top(), rect.right(), rect.bottom()
    # draw face bounding box
    cv2.rectangle(img, (x,y),(x+w, y+h), (0,255,0), 3)

while True:

    cv2.imshow("Output", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
