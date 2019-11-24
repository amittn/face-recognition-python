# import the libraries
import os
from PIL import Image
import face_recognition

# make a list of all the available images
image_names = os.listdir('images')

# load your image
image_to_be_matched = face_recognition.load_image_file('test.jpg')

# encoded the loaded image into a feature vector
image_to_be_matched_encoded = face_recognition.face_encodings(
    image_to_be_matched)[0]

# iterate over each image
for image in image_names:
    # load the image
    current_image = face_recognition.load_image_file("images/" + image)
    # encode the loaded image into a feature vector
    current_image_encoded = face_recognition.face_encodings(current_image)[0]
    # match your image with the image and check if it matches
    result = face_recognition.compare_faces([current_image_encoded], image_to_be_matched_encoded, 0.2)
    # check if it was a match
    if result[0] == True:
        print("Matched: The test image matched with " + image)
    else:
        print("Not matched: The test image did not match " + image)
