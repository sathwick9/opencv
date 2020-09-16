import cv2

video = cv2.VideoCapture('videoplayback.mp4')

classifier_file = 'cars.xml'

#create car classification
car_tracker = cv2.CascadeClassifier(classifier_file)

while True:

    #reading the current frame
    (read_successful, frame) = video.read()

    #safe coading
    if read_successful:
        #must convert to greyscale
        greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect car
    cars = car_tracker.detectMultiScale(greyscaled_frame)

    #drawing rectangle over cars detected
    for(x , y, w, h) in cars:

    #they are stored in an array
    #(0,0,255) colour of rectangle 2 is size of rectangle
    #car2 = cars[2] #(cars stored in an array)
    #(x ,y , w, h) = car2
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)

   

    #diaplay the image with the face spotted
    cv2.imshow('Clever Programmer Car Detector',frame)

    #dont autoclose (waait here in the code and listen for a key)
    cv2.waitKey(1)

print("code copleted")
