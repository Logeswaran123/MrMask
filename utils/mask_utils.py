import math
import random
from typing import Any
import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

from utils.video_reader_utils import VideoReader
from utils.const import INDEX_FINGER_TIP, MIDDLE_FINGER_MCP, \
                        PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD, MASK, \
                        FULL_FACE_MESH_LANDMARKS, HALF_FACE_MESH_LANDMARKS, \
                        BEARD_FACE_MESH_LANDMARKS
from utils.drawing_utils import Draw

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
face_detection = mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=0.75,
                    min_tracking_confidence=0.75)
holistic = mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
hands = mp_hands.Hands(
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

class Mask():
    """ Toplevel class for masks """
    def __init__(self, filename: Any, method_face: str, mask: int, require_mesh: bool) -> None:
        self.video_reader = VideoReader(filename)
        self.method_face = method_face
        self.init_mask = mask >= 0
        self.mask = random.randint(0, len(MASK) - 1) if mask == -1 else mask
        self.require_mesh = require_mesh
        self.width = int(self.video_reader.get_frame_width())
        self.height = int(self.video_reader.get_frame_height())
        self.video_fps = self.video_reader.get_video_fps()
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.key_points = self.centroid = self.face_traingle = None
        self.mask_tracker = []
        self.draw = Draw(self.width, self.height)
        self.track_face_detection = []

    def get_face_keypoints(self, image, face_landmarks):
        """ Get keypoints """
        key_points = {}
        image_rows, image_cols, _ = image.shape
        for idx, landmark in enumerate(face_landmarks.landmark):
            if ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
                continue
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                            image_cols, image_rows)
            if landmark_px:
                key_points[idx] = landmark_px
        return key_points

    def get_centroid(self):
        """ Get centroid of face mesh """
        points = list(self.key_points.values())
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        self.centroid = (int(sum(x) / len(points)), int(sum(y) / len(points)))

    def _check_point_in_triangle(self, point, a_pt, b_pt, c_pt):
        def area(x, y, z):
            return abs((x[0] * (z[1] - z[1]) + y[0] * (z[1] - x[1]) \
                        + z[0] * (x[1] - y[1])) / 2.0)

        A = area (a_pt, b_pt, c_pt) # Calculate area of triangle ABC
        A1 = area (point,  b_pt, c_pt)  # Calculate area of triangle PBC
        A2 = area (a_pt, point, c_pt)  # Calculate area of triangle PAC
        A3 = area (a_pt, b_pt, point)  # Calculate area of triangle PAB
        return A == A1 + A2 + A3

    def _bbox(self, image, results):
        """ Bounding Box method """
        detections = []
        if results.detections:
            for detection in results.detections:
                detections.append(detection)
                if self.require_mesh:
                    mp_drawing.draw_detection(image, detection)
        return (image, detections)

    def _overlay_mask(self, image, face_landmarks, bbox):
        """ Overlay method """
        def get_mesh_points(key_points):
            full_face_mesh_points = []
            half_face_mesh_points = []
            beard_face_mesh_points = []
            try:
                for point in FULL_FACE_MESH_LANDMARKS:
                    full_face_mesh_points.append(key_points[point])
                for point in HALF_FACE_MESH_LANDMARKS:
                    half_face_mesh_points.append(key_points[point])
                for point in BEARD_FACE_MESH_LANDMARKS:
                    beard_face_mesh_points.append(key_points[point])
            except:
                pass
            return (full_face_mesh_points, half_face_mesh_points, beard_face_mesh_points)

        self.key_points = self.get_face_keypoints(image, face_landmarks)
        self.get_centroid()
        face_width, face_height = _normalized_to_pixel_coordinates(bbox.width, bbox.height,
                                        image.shape[1], image.shape[0])
        mesh_points = get_mesh_points(self.key_points)
        image = self.draw.overlay_mask(image, face_width, face_height, \
                            self.centroid, self.mask, mesh_points)
        if self.require_mesh:
            image = self.draw.mediapipe_draw(image, face_landmarks, self.method_face)
        return image

    def _check_mask(self, index_finger_pt):
        """ Get mask ID """
        def dist_xy(point1, point2):
            """ Euclidean distance between two points point1, point2 """
            diff_point1 = (point1[0] - point2[0]) ** 2
            diff_point2 = (point1[1] - point2[1]) ** 2
            return (diff_point1 + diff_point2) ** 0.5

        mask_num = None
        dist = math.inf
        num_masks = len(MASK)
        for mask_idx in range(0 , num_masks):
            center_width = ((self.width // num_masks) - (self.width // num_masks) // 2) + \
                                (self.width // num_masks) * (mask_idx)
            center_height = (self.height // 8) // 2
            mask_center = (center_width, center_height)
            dist_mask_index = dist_xy(mask_center, index_finger_pt)
            if 0 <= dist_mask_index < dist and \
                    index_finger_pt[1] < (self.height // 8):
                dist = dist_mask_index
                mask_num = mask_idx
        return mask_num

    def _choose_mask(self, image, result_hands):
        """ Choose mask to overlay on face in image """
        index_finger_pt = None
        image_rows, image_cols, _ = image.shape
        if result_hands.multi_hand_landmarks:
            for hand_landmarks in result_hands.multi_hand_landmarks:
                landmark = hand_landmarks.landmark[INDEX_FINGER_TIP]
                index_finger_pt = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                            image_cols, image_rows)
                break

        image = self.draw.overlay_bar(image)
        if index_finger_pt is not None:
            mask_idx = self._check_mask(index_finger_pt)
            self.mask_tracker.append(mask_idx if mask_idx is not None else self.mask)

        if len(self.mask_tracker) == 12:
            if len(set(self.mask_tracker)) == 1:
                self.mask = self.mask_tracker[11]
            del self.mask_tracker[0]

        return image

    def _hand(self, image, results):
        """ Draw hands """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        return image

    def single_face(self, image, result_hands):
        """ Single face method """
        results_bbox = face_detection.process(image)
        results_mesh = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image, detections = self._bbox(image, results_bbox)
        if results_mesh.face_landmarks and len(detections) == 1:
            self.track_face_detection.append(True)
            bbox = detections[0].location_data.relative_bounding_box
            image = self._overlay_mask(image, results_mesh.face_landmarks, bbox)
        else:
            self.track_face_detection.append(False)

        if len(self.track_face_detection) == 5:
            if self.video_reader._is_live_cam and not all(self.track_face_detection) \
                    and len(set(self.track_face_detection)) == 1:
                self.mask = random.randint(0, len(MASK) - 1)
            del self.track_face_detection[0]

        if result_hands is not None:
            image = self._hand(image, result_hands)
        return image

    def multi_face(self, image, result_hands):
        """ Multi face method """
        detections = []
        results_bbox = face_detection.process(image)
        results_mesh = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image, detections = self._bbox(image, results_bbox)
        if results_mesh.multi_face_landmarks and \
                (len(detections) == len(results_mesh.multi_face_landmarks)):
            for face_idx, face_landmarks in enumerate(results_mesh.multi_face_landmarks):
                bbox = detections[face_idx].location_data.relative_bounding_box
                image = self._overlay_mask(image, face_landmarks, bbox)
        if result_hands is not None:
            image = self._hand(image, result_hands)
        return image

    def run(self) -> None:
        """ Run face mask """
        if self.video_reader.is_opened() is False:
            print("Error File Not Found.")

        out = cv2.VideoWriter("output.avi", self.fourcc, self.video_fps, (self.width, self.height))
        while self.video_reader.is_opened():
            result_hands = None
            image = self.video_reader.read_frame()
            if image is None:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.video_reader._is_live_cam and not self.init_mask:
                result_hands = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                image = self._choose_mask(image, result_hands)
            if self.method_face.lower() == "single":
                image = self.single_face(image, result_hands)
            else: # self.method_face.lower() == "multi"
                image = self.multi_face(image, result_hands)

            # Flip the image horizontally for a selfie-view display.
            if self.video_reader._is_live_cam:
                image = cv2.flip(image, 1)
            out.write(image)
            cv2.imshow('MediaPipe Face Mesh', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release()
