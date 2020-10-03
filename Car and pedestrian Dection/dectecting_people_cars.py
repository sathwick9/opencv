import cv2

video = cv2.VideoCapture('car-and-pedestrian-video0.mp4')

car_tracker_file = 'cars.xml'
pedestrian_tracker = 'haarcascade_fullbody.xml'

#create car classification
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker)



while True:

    #reading the current frame
    (read_successful, frame) = video.read()

    #safe coading
    if read_successful:
        #must convert to greyscale
        greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect car and pedestrains
    cars = car_tracker.detectMultiScale(greyscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(greyscaled_frame)

    #drawing rectangle over cars detected
    

    #they are stored in an array
    #(0,0,255) colour of rectangle 2 is size of rectangle
    #car2 = cars[2] #(cars stored in an array)
    #(x ,y , w, h) = car2
    for(x , y, w, h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)

    for(x , y, w, h) in pedestrians:
         cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255),2) #chainging color to yello for pedestrians


    #diaplay the image with the face spotted
    cv2.imshow('Clever Programmer Car Detector',frame)

    #dont autoclose (waait here in the code and listen for a key)
    key = cv2.waitKey(1)

    #stop if q key is pressed
    if key == 81 or key == 113:
        break

#release the videocapture object
video.release()

print("code copleted")
