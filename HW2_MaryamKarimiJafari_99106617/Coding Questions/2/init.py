import cv2
import imutils
import numpy as np
from keras.models import load_model
# from playsound import playsound
# import keyboard

#def move_the_head():

lastperd=np.array([21])
pred=np.array([23])

# get the reference to the webcam
CAMERA = cv2.VideoCapture(0)

CAPTURE_WIDTH = 900
ROI_LONG = 500 # Region Of Interest
MARGIN = 50
TOP = MARGIN
RIGHT = CAPTURE_WIDTH - MARGIN
BOTTOM = TOP + ROI_LONG
LEFT = RIGHT - ROI_LONG

model = load_model('model_HW2_2_3.h5') #FER-2013_model_original

while(True):
    _, frame = CAMERA.read()
    frame = imutils.resize(frame, CAPTURE_WIDTH)
    # frame = cv2.flip(frame, 1)
    (height, width) = frame.shape[:2]

    # Add rectangle to original frame
    cv2.rectangle(frame, (LEFT, TOP), (RIGHT, BOTTOM), (0,255,0), 2)

    # Cut ROI and preprocess
    roi = frame[TOP+2:BOTTOM-2, LEFT+2:RIGHT-2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) # need fixes
    cv2.imshow("ROI", gray)

    # Predict and show prediction
    gray_small = imutils.resize(gray, 48)
    gray_small = gray_small.reshape(1,48,48,1)
    pred=model.predict(gray_small)
    pred=np.argmax(pred,axis=1)
    
    my_dict = {'[0]': 'angry', '[1]': 'disgust', '[2]': 'fear', '[3]': 'happy', '[4]': 'neutral', '[5]': 'sad', '[6]': 'surprise'}

    print("{}".format(my_dict[str(pred)]))

    # if(keyboard.is_pressed('g')):#(pred!=lastperd):
    #     try:
    #         playsound('{}.wav'.format(pred))
    #         #move()
    #         lastperd=pred
    #     except:
    #         pass

    # pred = model.predict_classes(gray_small)[0]
    LABEL_TEXT = my_dict[str(pred)]
    LABEL_COLOR = (0,255,0)
    cv2.putText(frame, LABEL_TEXT, (LEFT, TOP-7), cv2.FONT_HERSHEY_SIMPLEX, 1, LABEL_COLOR, 2)
    cv2.imshow("Frame", frame)

    # if the user pressed "q", then stop looping
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
