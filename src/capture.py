import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class KeypointExtractor:
    def __init__(self, mode=True, maxHands=2, detectionCon=0.2, trackCon=0.2):
        # Activar static_image_mode para detección en cada frame
        self.hands = mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            model_complexity=1,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.face = mp_face.FaceMesh(
            static_image_mode=mode,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.maxHands = maxHands



    def extract(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_hands = self.hands.process(image)
        res_face = self.face.process(image)
        if res_hands.multi_hand_landmarks:
            for hand_landmarks in res_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        if res_face.multi_face_landmarks:
            for face_landmarks in res_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        kp = []


        # Manos: slots fijos [Right, Left] con bit de presencia (x, y, z, p)
        hand_sides = {}
        if res_hands.multi_hand_landmarks and res_hands.multi_handedness:
            for lm_list, hd in zip(res_hands.multi_hand_landmarks, res_hands.multi_handedness):
                label = hd.classification[0].label
                hand_sides[label] = lm_list
        for side in ['Right', 'Left']:
            lm_list = hand_sides.get(side)
            if lm_list:
                for lm in lm_list.landmark:
                    kp.extend([lm.x, lm.y, lm.z, 1])
            else:
                kp.extend([0.0, 0.0, 0.0, 0] * 21)



        # Cara
        if res_face.multi_face_landmarks:
            face_lm = res_face.multi_face_landmarks[0]
            for lm in face_lm.landmark:
                kp.extend([lm.x, lm.y, lm.z])
        else:
            kp.extend([0] * (468 * 3))
        return np.array(kp)
