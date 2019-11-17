import face_recognition
import argparse
import os
from PIL import Image


group1_images = os.listdir('group1')
group2_images = os.listdir('group2')

print(len(group1_images))
print(len(group2_images))


def do_stuff():
    print(len(group1_images))
    print(len(group2_images))

    # iterate over each group1 image
    for group1_image in group1_images:
        image_to_be_matched = face_recognition.load_image_file("group1/" + group1_image)
        image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]

        # iterate over each group2 image
        for group2_image in group2_images:
            # load the image
            current_image = face_recognition.load_image_file("group2/" + group2_image)
            # encode the loaded image into a feature vector
            current_image_encoded = face_recognition.face_encodings(current_image)[0]
            # match your image with the image and check if it matches
            result = face_recognition.compare_faces([image_to_be_matched_encoded], current_image_encoded)

            # check if it was a match
            if result[0] == True:
                print("Matched: " + group1_image + "with " + group2_image)
            else:
                print("Not matched: ")



def pull_faces(folder, group_image):
    face_locations = face_recognition.face_locations(group_image)

    for face_location in face_locations:
        top, right, bottom, left = face_location

        face_image = group_image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        #pil_image.show(pil_image)
        pil_image.save(f'{folder}{top}.jpg')

    return face_locations


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i_1", "--image_1", required=True, help="path to the image_1 file")
    ap.add_argument("-i_2", "--image_2", required=True, help="path to the image_2 file")
    args = vars(ap.parse_args())
    return args


if __name__ == "__main__":
    args = parse_arguments()
    image1 = face_recognition.load_image_file(args['image_1'])
    pull_faces("group1/", image1)

    image2 = face_recognition.load_image_file(args['image_2'])
    pull_faces("group2/", image2)

    do_stuff()
