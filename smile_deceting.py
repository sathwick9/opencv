import cv2

#face ans smile classifiers

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

#grab web cam feed
webcam = cv2.VideoCapture(0)

while True:

    #read current frame from webcam
    successful_frame_read, frame = webcam.read()

    #If there's an error , abort
    if not successful_frame_read:
        break

    #change to greyscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #first dectect faces
    faces = face_detector.detectMultiScale(frame_grayscale) #, 1.3, 5)

    #run smile detector with each faces
    for (x, y, w, h) in faces:

        #draw a rectangle around the face
        cv2.rectangle(frame,(x, y),(x+w, y+h), (100, 200, 50), 4)

        #creat the face sub-image (opencv allows you to subindex like this.
        #its built on numpy),slice a m-dimension array)
        the_face = frame[y:y+h, x:x+w]

        #greyscale the faces
        face_grayscale =cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        #detect smile in the faces
        smile = smile_detector.detectMultiScale(face_grayscale,
        scaleFactor=1.7, minNeighbors=20)

        #find all the smilires in the face
        #for(x_, y_, w_ ,h_) in smile:

            #draw a rectangle around the smile
            #cv2.rectangle(the_face, (x_,y_),(x_ + w_ , y_ + h_), (50,50,200),4)

        #drawing a rectangle over smile
        #label this face as smile
        if len(smile) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
            fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    #show the current frame
    cv2.imshow('Smile Deteqctor', frame)

    #dont autoclose (waait here in the code and listen for a key)
    key = cv2.waitKey(1)

    #stop if q key is pressed
    if key == 81 or key == 113:
            break
#clean up
webcam.release()
cv2.destroyAllWindows()
