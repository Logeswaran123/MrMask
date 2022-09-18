import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from utils.const import PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD, MASK

RED_COLOR = (0, 0, 255)
THICKNESS = 5

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

class Draw():
    """ Helper class for drawing utilities """
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height

    def plot_landmarks(self, landmark_list: landmark_pb2.NormalizedLandmarkList,
                        elevation: int = 10, azimuth: int = 10):
        """ 3D plot landmarks of face mesh """
        def _normalize_color(color):
            return tuple(v / 255. for v in color)

        if not landmark_list:
            return
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.view_init(elev=elevation, azim=azimuth)
        plotted_landmarks = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and
                landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < PRESENCE_THRESHOLD)):
                continue
            ax.scatter3D(
                xs=[-landmark.z],
                ys=[landmark.x],
                zs=[-landmark.y],
                color=_normalize_color(RED_COLOR[::-1]),
                linewidth=THICKNESS)
            plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
        plt.show()

    def overlay_mask(self, image, face_width, face_height, centroid, mask_idx, mesh_points):
        """ Overlay mask in image """
        # # Method 1: Using bounding box coordinates
        # copy = image.copy()
        # file_name = MASK[mask_idx]
        # mask = cv2.imread("./data/masks/" + file_name)
        # mask = cv2.resize(mask, (face_width + face_width // 2, face_height + face_height // 2))
        # mask_h, mask_w, _ = mask.shape
        # # Overlay mask
        # try:
        #     copy[int(centroid[1] - mask_h / 2) : int(centroid[1] + mask_h / 2), \
        #             int(centroid[0] - mask_w / 2) : int(centroid[0] + mask_w / 2)] = mask
        # except:
        #     pass
        # return copy

        # Method 2: Using face mesh coordinates
        # Refer MESH_LANDMARKS in utils/const.py for landmarks used.
        file_name = MASK[mask_idx]
        mask_annotation = "./data/masks/" + file_name[:-4] + ".csv"
        with open(mask_annotation) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            csv_reader = sorted(csv_reader, key=lambda row: row[0], reverse=False)
            mask_pts = []
            for _, row in enumerate(csv_reader):
                # skip head or empty line if it's there
                try:
                    mask_pts.append(np.array([float(row[1]), float(row[2])]))
                except ValueError:
                    continue

        if len(mask_pts) == len(mesh_points[0]):
            mesh_points = mesh_points[0]
        elif len(mask_pts) == len(mesh_points[1]):
            mesh_points = mesh_points[1]
        else:
            mesh_points = mesh_points[2]
        if len(mask_pts) != len(mesh_points):
            return image

        src_pts = np.array(mask_pts, dtype="float32")
        dst_pts = np.array(mesh_points, dtype="float32")
        mask_img = cv2.imread("./data/masks/" + file_name, cv2.IMREAD_UNCHANGED)
        mask_img = mask_img.astype(np.float32) / 255.0

        copy = image.copy()
        copy = copy.astype(np.float32) / 255.0

        # Get the perspective transformation matrix
        M, _ = cv2.findHomography(src_pts, dst_pts)
        # Transformed masked image
        transformed_mask = cv2.warpPerspective(
            mask_img,
            M,
            (copy.shape[1], copy.shape[0]),
            None,
            cv2.INTER_LINEAR,
            cv2.BORDER_CONSTANT,
        )

        # Overlay mask
        alpha_mask = transformed_mask[:, :, 3]
        alpha_image = 1.0 - alpha_mask
        for c in range(0, 3):
            copy[:, :, c] = (
                alpha_mask * transformed_mask[:, :, c]
                + alpha_image * copy[:, :, c]
            )
        copy = (copy * 255).astype(np.uint8)
        return copy

    def overlay_bar(self, image):
        """ Overlay horiontal bar in image """
        def display_mask_on_bar(image, mask_width, mask_height, mask_center, mask_idx):
            # # Method 1: Using coordinates and overwriting pixels directly
            # copy = image.copy()
            # file_name = MASK[mask_idx]
            # mask = cv2.imread("./data/masks/" + file_name)
            # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            # mask = cv2.resize(mask, (mask_width + mask_width // 2, mask_height + mask_height // 2))
            # mask_h, mask_w, _ = mask.shape

            # # Overlay mask
            # try:
            #     copy[int(mask_center[1] - mask_h / 2) : int(mask_center[1] + mask_h / 2), \
            #             int(mask_center[0] - mask_w / 2) : int(mask_center[0] + mask_w / 2)] = mask
            # except:
            #     pass
            # return copy

            # Method 2: Using alpha channel for transparent mask overlay
            # Source: https://stackoverflow.com/a/71701023
            copy = image.copy()
            file_name = MASK[mask_idx]
            mask = cv2.imread("./data/masks/" + file_name, cv2.IMREAD_UNCHANGED)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2BGRA)
            mask = cv2.resize(mask, (mask_width + mask_width // 2, mask_height + mask_height // 2))
            mask_h, mask_w, _ = mask.shape
            alpha_channel = mask[:, :, 3] / 255
            overlay_colors = mask[:, :, :3]
            alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
            background_subsection = copy[int(mask_center[1] - mask_h / 2) : int(mask_center[1] + mask_h / 2), \
                                            int(mask_center[0] - mask_w / 2) : int(mask_center[0] + mask_w / 2)]
            composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask
            try:
                copy[int(mask_center[1] - mask_h / 2) : int(mask_center[1] + mask_h / 2), \
                            int(mask_center[0] - mask_w / 2) : int(mask_center[0] + mask_w / 2)] = composite
            except:
                pass
            return copy

        alpha = 0.5
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height // 8) , (25,25,25), -1)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        num_masks = len(MASK)
        mask_width = mask_height = self.height // 8 - ((self.height // 8) // 2)
        for mask_idx in range(0 , num_masks):
            center_width = ((self.width // num_masks) - (self.width // num_masks) // 2) + \
                                (self.width // num_masks) * (mask_idx)
            center_height = (self.height // 8) // 2
            mask_center = (center_width, center_height)
            image = display_mask_on_bar(image, mask_width, mask_height, mask_center, mask_idx)
        return image

    def mediapipe_draw(self, image, face_landmarks, method_face):
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        if method_face.lower() == "multi":
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
        return image
