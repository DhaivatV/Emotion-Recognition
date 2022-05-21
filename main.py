import cv2
from cv2 import imshow
from emotion_recognition import response

model =  r'/home/dhaivat/emotion-recognition/model.h5'

test_img = cv2.imread('path to image of which emotion is to be predicted')
test_img = cv2.resize(test_img, (48,48))
test_img = test_img.reshape(1,48,48,1)
pred_img = model.predict(test_img)
respons = response(pred_img)



