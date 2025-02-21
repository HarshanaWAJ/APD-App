import pickle
import cv2
import mediapipe as mp
import numpy as np

def test_model():
    # Load the trained model
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

    # Labels dictionary, update with your actual labels if needed
    labels_dict = {0: 'A',
                   1: 'B',
                   2: 'C',
                   3: 'D',
                   4: 'E'}

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        H, W, _ = frame.shape

        # Convert frame to RGB for hand landmark processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and extract hand landmarks
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    # Extract normalized x and y coordinates
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize the coordinates so that they are in the range [0, 1]
                # Append normalized x and y values (this ensures we get 42 features total)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Check if the feature vector length is 42
            print(f"Number of features: {len(data_aux)}")  # Should print 42

            if len(data_aux) != 42:
                print("Error: Feature count mismatch.")
                continue  # Skip the frame if the feature count is incorrect

            # Predict using the model
            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]  # Translate the prediction to a label

            # Draw bounding box around the hand and put the predicted label
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)

        # Display the frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
