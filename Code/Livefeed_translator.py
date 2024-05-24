import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

'''
    NOTE: This is a live-feed translation program that can be used IF
    the images have been preprocessed using Reshape_all.py if other
    preprocessing methods have been used, this needs to be altered aswell.
'''

mp_drawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands
path_model = 'C:\\Users\\theoj\\Desktop\\KandidatUpsatsProject\\Exam_project\\checkpoint_mobilenet_lr=0,0001_epochs=4'
model = tf.keras.models.load_model(path_model)
model.summary()


labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
target_width = 224
target_height = 224
frame_skip = 0

cap = cv2.VideoCapture(0)

# Confidence level and number of hands
hands = mphands.Hands(min_detection_confidence=0.5, max_num_hands=1)

# Threshold size for filtering out detections based on hand size
threshold_size = 1000

# Bounding box offset
offset = 50

while True:
    data, image = cap.read()
    # Flip
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # Store
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find hand
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Calculate hand bounding box
        landmarks_as_tuples = [(int(l.x * image.shape[1]), int(l.y * image.shape[0])) for l in hand_landmarks.landmark]
        bbox = cv2.boundingRect(np.array(landmarks_as_tuples))

        # Add offset to the bounding box
        bbox = (bbox[0] - offset, bbox[1] - offset, bbox[2] + 2 * offset, bbox[3] + 2 * offset)

        # Filter out detections based on hand size
        if bbox[2] * bbox[3] < threshold_size:
            continue

        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 0), 2)

        # Cropping and displaying image
        cropped_image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        frame_skip+=1
        if cropped_image is not None and cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
            if frame_skip > 20:
                original_height, original_width, _ = cropped_image.shape

                # Calculate scaling factors
                scale_x = target_width / original_width
                scale_y = target_height / original_height

                # Resize the image to fit within the canvas
                if scale_x < scale_y:
                    new_width = target_width
                    new_height = int(original_height * scale_x)
                else:
                    new_width = int(original_width * scale_y)
                    new_height = target_height

                resized_img = cv2.resize(cropped_image, (new_width, new_height))

                # Create a white canvas of target size
                black_canvas = 0 * np.ones((target_height, target_width, 3), dtype=np.uint8)

                # Calculate offset to center the resized image on the canvas
                offset_x = (target_width - new_width) // 2
                offset_y = (target_height - new_height) // 2

                # Paste the resized image onto the black canvas
                black_canvas[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = resized_img

                # Check if the cropped image has valid dimensions
                if black_canvas.shape[0] > 0 and black_canvas.shape[1] > 0:
                    # Convert black_canvas to float32 and rescale pixel values to [0, 1]
                    black_canvas_batch = black_canvas.astype(np.float32) / 255.0
                    black_canvas_batch = np.expand_dims(black_canvas_batch, axis=0)
                    prediction = model.predict(black_canvas_batch, verbose=0)
                    prediction = np.round(prediction,3)
                    predicted_index = np.argmax(prediction, axis=1)
                    # Get the indices of the top three predictions
                    top_indices = np.argsort(prediction[0])[::-1][:3]

                    # Iterate over the top three predictions
                    for i, index in enumerate(top_indices):
                        # Get the label and percentage of the current prediction
                        label = labels[index]
                        percentage = prediction[0][index]

                        # Check if the percentage is above 0.1
                        if percentage > 0.0:
                            # Format the prediction text
                            prediction_text = f'{label}: {percentage:.2f}'

                            # Calculate the position to display the prediction text in the top right corner
                            text_size = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                            text_x = image.shape[1] - text_size[0] - 10
                            text_y = 30 + i * 30  # Adjust vertical spacing

                            # Draw a rectangle as background for the prediction text
                            cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 10, text_y + 5), (0, 0, 0), -1)

                            # Draw the prediction text
                            cv2.putText(image, prediction_text, (text_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.putText(image, 'Quit: Press ''q''', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Handtracker', image)
    # Hotkey for "quit" -> q
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

