# Copyright (c) 2025 Oguzhan Cagirir (OguzhanCOG), KCL Electronics Society
#
# Project: KCL FoAI RoboFootball System
# File: setup.py
# Description: Utility script for camera calibration and calculating perspective transformation matrices for the arena.
#
# Author: Oguzhan Cagirir (OguzhanCOG)
# Date: May 24, 2025
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Open Source Initiative.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
#
# You should have received a copy of the MIT License
# along with this program. If not, see <https://opensource.org/licenses/MIT>.
#
# --- Version: 1.2.0 ---

import cv2
import pupil_apriltags
import numpy as np
import time
import os
import glob 

CAMERA_INDEX = 1
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 960

CHECKERBOARD_INTERNAL_CORNERS_WIDTH = 8
CHECKERBOARD_INTERNAL_CORNERS_HEIGHT = 6
SQUARE_SIZE_MM = 30.0
MIN_IMAGES_FOR_CALIBRATION = 15

APRILTAG_FAMILY = "tag36h11"
TAG_SIZE_METERS = 0.093

SIM_ARENA_WIDTH_M = 2.4
SIM_ARENA_HEIGHT_M = 2.0

CORNER_TAG_DEFINITIONS = { 
    1: ("PTag1 (Sim Top-Left)",       (0.0, 0.0)),
    3: ("PTag3 (Sim Top-Right)",      (SIM_ARENA_WIDTH_M, 0.0)),
    2: ("PTag2 (Sim Bottom-Right)",   (SIM_ARENA_WIDTH_M, SIM_ARENA_HEIGHT_M)),
    0: ("PTag0 (Sim Bottom-Left)",    (0.0, SIM_ARENA_HEIGHT_M)),
}
REQUIRED_CORNER_TAG_IDS = sorted(list(CORNER_TAG_DEFINITIONS.keys()))

TAG_ID_MAP_GENERAL = {tag_id: details[0] for tag_id, details in CORNER_TAG_DEFINITIONS.items()}
TAG_ID_MAP_GENERAL.update({4: "Robot B1", 5: "Robot B2"}) 

CALIBRATION_DATA_FOLDER = "camera_calibration_data"
PERSPECTIVE_MATRIX_FILE_NPZ = "perspective_transform_matrices.npz" 
CALIBRATION_FILE_NAME_TEMPLATE = "camera_calibration_{width}x{height}.npz"

DETECTOR_KWARGS = {
    'families': APRILTAG_FAMILY, 'nthreads': 1, 'quad_decimate': 1.0,
    'quad_sigma': 0.0, 'refine_edges': True, 'decode_sharpening': 0.25, 'debug': False
}

_printed_messages = set()
def print_once(message_key, message_content):
    if message_key not in _printed_messages: print(message_content); _printed_messages.add(message_key)

def get_user_choice(prompt, default_yes=True):
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        choice = input(f"{prompt} {suffix}: ").strip().lower()
        if not choice: return default_yes
        if choice == 'y': return True
        if choice == 'n': return False
        print("Invalid input. Please enter 'y' or 'n'.")

def perform_camera_calibration(cap, current_width, current_height):

    print("\n--- Starting New Camera Calibration ---")
    objp = np.zeros((CHECKERBOARD_INTERNAL_CORNERS_HEIGHT * CHECKERBOARD_INTERNAL_CORNERS_WIDTH, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_INTERNAL_CORNERS_WIDTH, 0:CHECKERBOARD_INTERNAL_CORNERS_HEIGHT].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_MM
    objpoints, imgpoints = [], []
    img_count = 0
    last_capture_time = time.time()

    print(f"Instructions for Calibration Image Capture (Resolution: {current_width}x{current_height}):")
    print("  Hold checkerboard flat, visible. Vary angles, distances, positions.")
    print("  Press 'c' to capture, 'q' to finish & calibrate, 'ESC' to skip.")

    while True:
        ret, frame = cap.read()
        if not ret: print_once("calib_frame_fail", "ERROR: Failed to grab frame during calibration."); time.sleep(0.1); continue
        if frame.shape[1] != current_width or frame.shape[0] != current_height:
            print_once("calib_res_skip", f"Calib: Frame res {frame.shape[1]}x{frame.shape[0]} != expected {current_width}x{current_height}. Skipping."); time.sleep(0.1); continue

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD_INTERNAL_CORNERS_WIDTH, CHECKERBOARD_INTERNAL_CORNERS_HEIGHT), None)
        key = cv2.waitKey(1) & 0xFF

        if ret_corners:
            cv2.putText(display_frame, "Corners Detected! Press 'c' to capture.", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display_frame, (CHECKERBOARD_INTERNAL_CORNERS_WIDTH, CHECKERBOARD_INTERNAL_CORNERS_HEIGHT), corners_subpix, ret_corners)
            if key == ord('c') and (time.time() - last_capture_time > 0.5):
                objpoints.append(objp)
                imgpoints.append(corners_subpix)
                img_count += 1; last_capture_time = time.time()
                img_filename = os.path.join(CALIBRATION_DATA_FOLDER, f"calib_img_{img_count:03d}_{current_width}x{current_height}.png")
                cv2.imwrite(img_filename, frame)
                print(f"Captured calib image {img_count}: {img_filename} ({len(objpoints)} total)")
        else:
            cv2.putText(display_frame, "No corners detected.", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        cv2.putText(display_frame, f"Calib Images: {img_count}/{MIN_IMAGES_FOR_CALIBRATION}", (10,40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0),2)

        scaled_h, scaled_w = display_frame.shape[0], display_frame.shape[1]
        if scaled_h > 720: display_frame = cv2.resize(display_frame, (int(scaled_w*720/scaled_h), 720))
        cv2.imshow('Camera Calibration - Image Capture', display_frame)

        if key == ord('q'):
            if img_count < 5: print("WARNING: Not enough images for reliable calibration.")
            else: print("Finishing calibration image capture..."); break
        elif key == 27: print("Skipping camera calibration."); return None, None 
    cv2.destroyWindow('Camera Calibration - Image Capture')
    if len(objpoints) > 4:
        print(f"Performing calibration with {len(objpoints)} images...")
        ret_calib, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (current_width, current_height), None, None)
        if ret_calib:
            print("Calibration Successful!")
            calib_file_name = CALIBRATION_FILE_NAME_TEMPLATE.format(width=current_width, height=current_height)
            calib_file_path = os.path.join(CALIBRATION_DATA_FOLDER, calib_file_name)
            np.savez(calib_file_path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
                     image_width=np.int32(current_width), image_height=np.int32(current_height)) 
            print(f"Calibration data saved to: {calib_file_path}")
            return camera_matrix, dist_coeffs
        else: print("ERROR: Camera calibration failed."); return None, None
    else: print("Not enough data for calibration."); return None, None

def calculate_and_save_all_perspective_transforms(cap, camera_matrix, dist_coeffs, actual_width, actual_height, detector):
    """Calculates and saves BOTH M_metric_to_sim and M_pixel_to_sim."""
    print("\n--- Setting Up ALL Perspective Transforms ---")
    print("Looking for required corner tags:", REQUIRED_CORNER_TAG_IDS)
    print("Position all 4 corner tags, then press 's' to capture points, 'q' to skip/quit.")

    fx, fy = camera_matrix[0,0], camera_matrix[1,1]
    cx, cy = camera_matrix[0,2], camera_matrix[1,2]
    camera_params_for_detector = (fx, fy, cx, cy)

    while True:
        ret, frame = cap.read()
        if not ret: print_once("pt_frame_fail","ERROR: Failed to grab frame for PT."); time.sleep(0.1); continue
        if frame.shape[1] != actual_width or frame.shape[0] != actual_height:
            print_once("pt_res_skip",f"PT: Frame res {frame.shape[1]}x{frame.shape[0]} != expected {actual_width}x{actual_height}. Skipping."); time.sleep(0.1); continue

        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)
        gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray_frame, True, camera_params_for_detector, TAG_SIZE_METERS)

        vis_frame = undistorted_frame.copy()
        current_corners_cam_metric_xy = {} 
        current_corners_img_pixel_xy = {}  

        for tag in detections:
            tag_id = tag.tag_id
            corners_px_vis = np.round(tag.corners).astype(int) 
            display_name = TAG_ID_MAP_GENERAL.get(tag_id, f"ID {tag.tag_id}")

            if tag_id in REQUIRED_CORNER_TAG_IDS:

                current_corners_cam_metric_xy[tag_id] = (tag.pose_t.flatten()[0], tag.pose_t.flatten()[1])

                current_corners_img_pixel_xy[tag_id] = (tag.center[0], tag.center[1])

                cv2.polylines(vis_frame, [corners_px_vis], True, (255,0,0), 3) 
                display_name = CORNER_TAG_DEFINITIONS[tag_id][0]
            else: 
                cv2.polylines(vis_frame, [corners_px_vis], True, (0,255,0), 2)

            text_pos = (corners_px_vis[0][0], corners_px_vis[0][1] - 15)
            cv2.putText(vis_frame, f"{display_name}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255),2)
            cv2.putText(vis_frame, f"Z:{tag.pose_t.flatten()[2]:.2f}m", (text_pos[0], text_pos[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1)

        all_found = all(tid in current_corners_cam_metric_xy for tid in REQUIRED_CORNER_TAG_IDS) 
        if all_found: cv2.putText(vis_frame, "All corners DETECTED! Press 's' to set.", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        else: cv2.putText(vis_frame, f"Missing: {[tid for tid in REQUIRED_CORNER_TAG_IDS if tid not in current_corners_cam_metric_xy]}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        scaled_h, scaled_w = vis_frame.shape[0], vis_frame.shape[1]
        if scaled_h > 720 : vis_frame = cv2.resize(vis_frame, (int(scaled_w*720/scaled_h), 720))
        cv2.imshow('Perspective Transform Setup', vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("Skipping perspective transform calculation."); return None, None 
        elif key == ord('s'):
            if all_found:
                print("Capturing points for BOTH perspective transforms...")
                src_metric_list, dst_sim_list_metric = [], []
                src_pixel_list, dst_sim_list_pixel = [], [] 

                for tid_phys in REQUIRED_CORNER_TAG_IDS: 

                    metric_x, metric_y = current_corners_cam_metric_xy[tid_phys]
                    src_metric_list.append([metric_x, metric_y])
                    dst_sim_list_metric.append(list(CORNER_TAG_DEFINITIONS[tid_phys][1]))
                    print(f"  Metric: {CORNER_TAG_DEFINITIONS[tid_phys][0]}: Cam_Metric({metric_x:.3f},{metric_y:.3f}) -> Sim{dst_sim_list_metric[-1]}")

                    pixel_x, pixel_y = current_corners_img_pixel_xy[tid_phys]
                    src_pixel_list.append([pixel_x, pixel_y])
                    dst_sim_list_pixel.append(list(CORNER_TAG_DEFINITIONS[tid_phys][1]))
                    print(f"  Pixel:  {CORNER_TAG_DEFINITIONS[tid_phys][0]}: Img_Pixel({pixel_x:.1f},{pixel_y:.1f}) -> Sim{dst_sim_list_pixel[-1]}")

                M_metric_to_sim = cv2.getPerspectiveTransform(np.float32(src_metric_list), np.float32(dst_sim_list_metric))
                M_pixel_to_sim = cv2.getPerspectiveTransform(np.float32(src_pixel_list), np.float32(dst_sim_list_pixel))

                print("\nPerspective Transformation Matrix (M_metric_to_sim - for AprilTag Poses):\n", M_metric_to_sim)
                print("\nPerspective Transformation Matrix (M_pixel_to_sim - for Ball Pixels):\n", M_pixel_to_sim)

                np.savez(PERSPECTIVE_MATRIX_FILE_NPZ, 
                         M_metric_to_sim=M_metric_to_sim, 
                         src_points_cam_metric_xy=np.float32(src_metric_list),
                         M_pixel_to_sim=M_pixel_to_sim,
                         src_points_img_pixel_xy=np.float32(src_pixel_list),
                         dst_points_sim_xy=np.float32(dst_sim_list_metric), 
                         sim_arena_width=SIM_ARENA_WIDTH_M,
                         sim_arena_height=SIM_ARENA_HEIGHT_M, 
                         corner_tag_definitions_used=CORNER_TAG_DEFINITIONS,
                         camera_resolution_when_captured=(actual_width, actual_height))
                print(f"\nTransformation matrices and reference points saved to: {PERSPECTIVE_MATRIX_FILE_NPZ}");
                cv2.destroyWindow('Perspective Transform Setup')
                return M_metric_to_sim, M_pixel_to_sim 
            else: print("Not all corners detected. Cannot set points.")
    return None, None 

def main_setup_flow():
    if not os.path.exists(CALIBRATION_DATA_FOLDER): os.makedirs(CALIBRATION_DATA_FOLDER)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): print(f"FATAL: Cannot open camera {CAMERA_INDEX}"); return

    print(f"Attempting to set initial camera resolution to: {DESIRED_WIDTH}x{DESIRED_HEIGHT}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    time.sleep(0.5)
    actual_width, actual_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera operational at: {actual_width}x{actual_height}")

    current_calib_file = os.path.join(CALIBRATION_DATA_FOLDER, CALIBRATION_FILE_NAME_TEMPLATE.format(width=actual_width, height=actual_height))
    camera_matrix, dist_coeffs = None, None

    if os.path.exists(current_calib_file) and get_user_choice(f"Found '{current_calib_file}'. Use it?", default_yes=True):
        try:
            data = np.load(current_calib_file); camera_matrix, dist_coeffs = data['camera_matrix'], data['dist_coeffs']
            stored_w, stored_h = int(data.get('image_width',0)), int(data.get('image_height',0))
            if stored_w > 0 and (stored_w != actual_width or stored_h != actual_height):
                print(f"WARNING: Loaded calib file for {stored_w}x{stored_h}, camera is {actual_width}x{actual_height}.")
                if not get_user_choice("Proceed with mismatched calibration?", default_yes=False): camera_matrix, dist_coeffs = None, None
            if camera_matrix is not None: print("Loaded existing calibration data.")
        except Exception as e: print(f"Error loading '{current_calib_file}': {e}. Will perform new calibration.")

    if camera_matrix is None or dist_coeffs is None:
        camera_matrix, dist_coeffs = perform_camera_calibration(cap, actual_width, actual_height)

    if camera_matrix is None or dist_coeffs is None:
        print("Camera calibration failed/skipped. Cannot proceed."); cap.release(); cv2.destroyAllWindows(); return

    final_fx, final_fy = camera_matrix[0,0], camera_matrix[1,1]
    final_cx, final_cy = camera_matrix[0,2], camera_matrix[1,2]

    detector = pupil_apriltags.Detector(**DETECTOR_KWARGS)
    M_metric, M_pixel = None, None 

    if os.path.exists(PERSPECTIVE_MATRIX_FILE_NPZ) and get_user_choice(f"Found '{PERSPECTIVE_MATRIX_FILE_NPZ}'. Use it?", default_yes=True):
        try:
            data = np.load(PERSPECTIVE_MATRIX_FILE_NPZ)
            M_metric = data['M_metric_to_sim']
            M_pixel = data['M_pixel_to_sim'] 
            stored_res = data.get('camera_resolution_when_captured')
            if stored_res is not None and (stored_res[0] != actual_width or stored_res[1] != actual_height):
                print(f"WARNING: Existing perspective matrices were for res {stored_res}, current is {actual_width}x{actual_height}.")
                if not get_user_choice("Continue with these matrices despite resolution mismatch?", default_yes=False): M_metric, M_pixel = None, None
            if M_metric is not None and M_pixel is not None: print("Loaded existing perspective transform matrices.")
        except Exception as e: print(f"Error loading '{PERSPECTIVE_MATRIX_FILE_NPZ}': {e}. Will calculate new ones.")

    if M_metric is None or M_pixel is None: 
        M_metric, M_pixel = calculate_and_save_all_perspective_transforms(cap, camera_matrix, dist_coeffs, actual_width, actual_height, detector)

    if M_metric is None or M_pixel is None: print("Perspective transform calculation failed/skipped.")
    else:
        print("\n--- Setup Complete ---")
        print(f"Camera calibration active for {actual_width}x{actual_height}.")
        print("Perspective transformation matrices (metric-to-sim and pixel-to-sim) active.")
        print("You are ready to use these in your main simulation!")

        if get_user_choice("Enter live test mode (uses M_metric_to_sim for robot tag)?", default_yes=True):
            print("\n--- Live Test Mode (Press 'q' to quit) ---")
            print("Move Robot B1 (Tag ID 4) into view.")

            test_cam_params = (camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2])
            while True:
                ret, frame = cap.read()
                if not ret: break
                if frame.shape[1]!=actual_width or frame.shape[0]!=actual_height: continue
                undistorted = cv2.undistort(frame,camera_matrix,dist_coeffs,None,camera_matrix)
                gray = cv2.cvtColor(undistorted,cv2.COLOR_BGR2GRAY)
                detections = detector.detect(gray,True,test_cam_params,TAG_SIZE_METERS)
                vis = undistorted.copy()
                for tag in detections:
                    cv2.polylines(vis,[np.round(tag.corners).astype(int)],True,(0,255,0),2)
                    cv2.putText(vis,TAG_ID_MAP_GENERAL.get(tag.tag_id,f"ID{tag.tag_id}"),tuple(np.round(tag.corners[0]).astype(int)-np.array([0,10])),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)
                    if tag.tag_id == 4: 
                        cam_x,cam_y=tag.pose_t.flatten()[0],tag.pose_t.flatten()[1]
                        pt_cam=np.float32([[ [cam_x,cam_y] ]])
                        pt_sim=cv2.perspectiveTransform(pt_cam,M_metric) 
                        if pt_sim is not None:
                            sim_x,sim_y=pt_sim[0][0][0],pt_sim[0][0][1]
                            txt=f"Robot B1(ID4): Sim({sim_x:.2f},{sim_y:.2f})m"
                            print(f"\r{txt}    ",end=""); cv2.putText(vis,txt,(10,60),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
                scaled_h,scaled_w=vis.shape[0],vis.shape[1]
                if scaled_h > 720 : vis = cv2.resize(vis, (int(scaled_w*720/scaled_h),720))
                cv2.imshow('Live Transformed Test',vis)
                if cv2.waitKey(1)&0xFF == ord('q'): break
            print()
    cap.release(); cv2.destroyAllWindows(); print("\nInitialization setup script finished.")

if __name__ == '__main__':
    main_setup_flow()