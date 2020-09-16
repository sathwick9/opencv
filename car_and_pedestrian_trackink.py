import cv2

#Image
img_file = 'traffic-jam-in-the-street-with-cars-on-a-highway-BMG2RB.jpg'

#pre trained car classifier
classifier_file = 'cars.xml'

#creating open cv image
img = cv2.imread(img_file)

#convert to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#create car classification
car_tracker = cv2.CascadeClassifier(classifier_file)

#detect car
cars = car_tracker.detectMultiScale(black_n_white)

#drawing rectangle over cars detected
for(x , y, w, h) in cars:

 #they are stored in an array
#(0,0,255) colour of rectangle 2 is size of rectangle
#car2 = cars[2] #(cars stored in an array)
#(x ,y , w, h) = car2
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),2)


#diaplay the image with the face spotted
cv2.imshow('Clever Programmer Car Detector',img)
#cv2.imshow('Clever Programmer Car Detector',black_n_white)
#it does not give output as square bcs it drew on colour not black and white



#dont autoclose (waait here in the code and listen for a key)
cv2.waitKey()


print ("code completed")