import cv2
import numpy as np
import mediapipe as mp
import os
import time


pictures_to_take_input = 10 # number of pictures to take for each letter
ori_folder_path = "directory" # folder to save the images to




labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z']
picture_count = 0
taken_pictures = 0
name = labels[0]
folder_path =ori_folder_path + name
files = os.listdir(ori_folder_path)
max_number_val = 0
pictures_to_take = 0

def letterPath(index=0):
    global pictures_to_take
    global pictures_to_take_input
    global picture_count
    global taken_pictures
    global name
    name = labels[index]
    global ori_folder_path
    global folder_path
    folder_path = ori_folder_path + name
    global files

    files = os.listdir(folder_path)
    max_number = 0
    for file_name in files:
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            try:
                number = int(file_name.split(name)[-1].split('.')[0])
                max_number = max(max_number, number)
            except ValueError:
                pass
    print(f"max numbers {max_number}")
    pictures_to_take = pictures_to_take_input + max_number
    picture_count = max_number
    taken_pictures = max_number + 1

    print(f'letter {labels[index]}')


letterPath(0)
print(name)
width, height = 75, 75
background_color = (0, 0, 0)
baseImage = np.full((height, width, 3), background_color, dtype=np.uint8)
CROPSIZE = 100
frame_picture_delay = 0
mp_drawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands
cap = cv2.VideoCapture(0)
hands = mphands.Hands(min_detection_confidence=0.5, max_num_hands=1)
threshold_size = 1000
offset = 50
start_time = time.time()
frame_count = 0
start_taking_pictures = False
letter_index = 0
while True:
    data, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks_as_tuples = [(int(l.x * image.shape[1]), int(l.y * image.shape[0])) for l in hand_landmarks.landmark]
        bbox = cv2.boundingRect(np.array(landmarks_as_tuples))
        bbox = (bbox[0] - offset, bbox[1] - offset, bbox[2] + 2 * offset, bbox[3] + 2 * offset)

        if bbox[2] * bbox[3] >= threshold_size:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        start_taking_pictures = True

    if start_taking_pictures:
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks_as_tuples = [(int(l.x * image.shape[1]), int(l.y * image.shape[0])) for l in
                                   hand_landmarks.landmark]
            bbox = cv2.boundingRect(np.array(landmarks_as_tuples))
            bbox = (bbox[0] - offset, bbox[1] - offset, bbox[2] + 2 * offset, bbox[3] + 2 * offset)
            if bbox[2] * bbox[3] >= threshold_size:
                cropped_image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                height, width = cropped_image.shape[:2]
                if width >= 75 and height >= 75:
                    re = cv2.resize(cropped_image, (CROPSIZE, CROPSIZE))
                    if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                        cv2.imshow("dd", re)
                        if frame_picture_delay == 10:

                            if picture_count < pictures_to_take:
                                cv2.imwrite(f'{folder_path}/{name}{taken_pictures}.jpg', cropped_image)

                                print(f'Training picture {taken_pictures} taken')
                                taken_pictures += 1
                            frame_picture_delay = 0
                            picture_count += 1
                        frame_picture_delay += 1

    cv2.putText(image, 'change letter, m,n', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'Quit: Press ''q''', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'Capture Right: Press ''c''', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.imshow('Handtracker', image)
    if (key == ord('m')):
        start_taking_pictures = False
        if (letter_index != len(labels) - 1):

            letter_index += 1
            letterPath(letter_index)
        print(labels[letter_index])
        print(name)
    if (key == ord('n')):
        start_taking_pictures = False

        if (letter_index != 0):

            letter_index -= 1
            letterPath(letter_index)
        print(labels[letter_index])
        print(name)

    if key == ord('q'):
        break





