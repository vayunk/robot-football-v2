# Copyright (c) 2025 Oguzhan Cagirir (OguzhanCOG), KCL Electronics Society
#
# Project: KCL FoAI RoboFootball System
# File: hsv_tuner.py
# Description: Utility script for interactively tuning HSV color thresholds for object (ball) detection using OpenCV.
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
# --- Version: 1.0.0 ---

import cv2
import numpy as np
import os
import time

CALIBRATION_DATA_FOLDER = "camera_calibration_data"

CAMERA_RESOLUTION_W = 1280
CAMERA_RESOLUTION_H = 960
CALIBRATION_FILE_NAME = f"camera_calibration_{CAMERA_RESOLUTION_W}x{CAMERA_RESOLUTION_H}.npz"
CALIBRATION_FILE_PATH = os.path.join(CALIBRATION_DATA_FOLDER, CALIBRATION_FILE_NAME)

CAMERA_INDEX = 1 

INITIAL_H_LOW = 20
INITIAL_S_LOW = 100
INITIAL_V_LOW = 100
INITIAL_H_HIGH = 35  
INITIAL_S_HIGH = 255
INITIAL_V_HIGH = 255

def nothing(x):
    pass

def main():

    camera_matrix, dist_coeffs = None, None
    try:
        calibration_data = np.load(CALIBRATION_FILE_PATH)
        camera_matrix = calibration_data['camera_matrix']
        dist_coeffs = calibration_data['dist_coeffs']

        calib_w = int(calibration_data.get('image_width', 0))
        calib_h = int(calibration_data.get('image_height', 0))
        if calib_w > 0 and (calib_w != CAMERA_RESOLUTION_W or calib_h != CAMERA_RESOLUTION_H):
            print(f"WARNING: Loaded calib file '{CALIBRATION_FILE_PATH}' is for {calib_w}x{calib_h}, "
                  f"but tuner is set for {CAMERA_RESOLUTION_W}x{CAMERA_RESOLUTION_H}. "
                  "Ensure consistency for best results when transferring values.")
        print(f"Successfully loaded camera calibration from: {CALIBRATION_FILE_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Calibration file not found at '{CALIBRATION_FILE_PATH}'.")
        print("       Proceeding without undistortion, which might affect tuning accuracy.")
    except Exception as e:
        print(f"ERROR loading calibration file '{CALIBRATION_FILE_PATH}': {e}")
        print("       Proceeding without undistortion.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera source: {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION_H)
    time.sleep(0.5) 

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened. Live resolution: {actual_width}x{actual_height}")
    if actual_width != CAMERA_RESOLUTION_W or actual_height != CAMERA_RESOLUTION_H:
        print(f"WARNING: Camera started at {actual_width}x{actual_height}, "
              f"but script expected {CAMERA_RESOLUTION_W}x{CAMERA_RESOLUTION_H}. "
              "Using actual resolution for display.")

    cv2.namedWindow("HSV Color Tuner")
    cv2.createTrackbar("H Low", "HSV Color Tuner", INITIAL_H_LOW, 179, nothing) 
    cv2.createTrackbar("S Low", "HSV Color Tuner", INITIAL_S_LOW, 255, nothing)
    cv2.createTrackbar("V Low", "HSV Color Tuner", INITIAL_V_LOW, 255, nothing)
    cv2.createTrackbar("H High", "HSV Color Tuner", INITIAL_H_HIGH, 179, nothing)
    cv2.createTrackbar("S High", "HSV Color Tuner", INITIAL_S_HIGH, 255, nothing)
    cv2.createTrackbar("V High", "HSV Color Tuner", INITIAL_V_HIGH, 255, nothing)

    cv2.namedWindow("Morphological Ops Control", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Erode Iter", "Morphological Ops Control", 1, 10, nothing)
    cv2.createTrackbar("Dilate Iter", "Morphological Ops Control", 1, 10, nothing)
    cv2.createTrackbar("Kernel Size", "Morphological Ops Control", 5, 15, nothing) 

    print("\n--- HSV Tuning Instructions ---")
    print("Place your YELLOW ball in the camera's view under typical arena lighting.")
    print("Adjust the sliders in the 'HSV Color Tuner' window until:")
    print("  - ONLY the ball appears as WHITE in the 'Mask' window.")
    print("  - Everything else should be BLACK.")
    print("Try to make the white ball region solid, with minimal noise.")
    print("Adjust 'Morphological Ops Control' sliders to clean the mask if needed.")
    print("Press 'p' to print the current HSV values to the console.")
    print("Press 'q' to quit.")
    print("=" * 40)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("ERROR: Can't receive frame. Exiting ...")
            break

        if frame_bgr.shape[1] != actual_width or frame_bgr.shape[0] != actual_height:
            print(f"Warning: Frame res changed to {frame_bgr.shape[1]}x{frame_bgr.shape[0]}. Trying to adapt.")
            actual_width, actual_height = frame_bgr.shape[1], frame_bgr.shape[0]

        if camera_matrix is not None and dist_coeffs is not None:
            frame_to_process = cv2.undistort(frame_bgr, camera_matrix, dist_coeffs, None, camera_matrix)
        else:
            frame_to_process = frame_bgr.copy()

        frame_hsv = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2HSV)

        h_low = cv2.getTrackbarPos("H Low", "HSV Color Tuner")
        s_low = cv2.getTrackbarPos("S Low", "HSV Color Tuner")
        v_low = cv2.getTrackbarPos("V Low", "HSV Color Tuner")
        h_high = cv2.getTrackbarPos("H High", "HSV Color Tuner")
        s_high = cv2.getTrackbarPos("S High", "HSV Color Tuner")
        v_high = cv2.getTrackbarPos("V High", "HSV Color Tuner")

        if h_high < h_low: 
            h_high = max(h_low + 1, h_high) 
            cv2.setTrackbarPos("H High", "HSV Color Tuner", h_high)

        lower_bound = np.array([h_low, s_low, v_low])
        upper_bound = np.array([h_high, s_high, v_high])

        mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)

        erode_iter = cv2.getTrackbarPos("Erode Iter", "Morphological Ops Control")
        dilate_iter = cv2.getTrackbarPos("Dilate Iter", "Morphological Ops Control")
        kernel_s = cv2.getTrackbarPos("Kernel Size", "Morphological Ops Control")
        if kernel_s % 2 == 0: kernel_s += 1 
        if kernel_s < 3 : kernel_s = 3 

        kernel = np.ones((kernel_s, kernel_s), np.uint8)

        processed_mask = mask.copy() 
        if erode_iter > 0:
            processed_mask = cv2.erode(processed_mask, kernel, iterations=erode_iter)
        if dilate_iter > 0:
            processed_mask = cv2.dilate(processed_mask, kernel, iterations=dilate_iter)

        display_scale_factor = 1.0
        if actual_height > 720: 
            display_scale_factor = 720.0 / actual_height

        def resize_for_display(img_to_resize):
            if display_scale_factor < 1.0:
                return cv2.resize(img_to_resize, (0,0), fx=display_scale_factor, fy=display_scale_factor)
            return img_to_resize

        cv2.imshow("Original (Processed BGR)", resize_for_display(frame_to_process))

        cv2.imshow("Raw Mask", resize_for_display(mask))
        cv2.imshow("Processed Mask (Final)", resize_for_display(processed_mask))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print("\n--- Current HSV Values ---")
            print(f"LOWER_YELLOW = np.array([{h_low}, {s_low}, {v_low}])")
            print(f"UPPER_YELLOW = np.array([{h_high}, {s_high}, {v_high}])")
            print(f"Erode Iter: {erode_iter}, Dilate Iter: {dilate_iter}, Kernel Size: {kernel_s}")
            print("-" * 26)

    cap.release()
    cv2.destroyAllWindows()
    print("\nHSV Tuner finished. Final printed values are your tuned parameters.")
    print(f"LOWER_YELLOW = np.array([{h_low}, {s_low}, {v_low}])")
    print(f"UPPER_YELLOW = np.array([{h_high}, {s_high}, {v_high}])")
    print(f"Morph Ops: Erode={erode_iter}, Dilate={dilate_iter}, Kernel={kernel_s}")

if __name__ == '__main__':
    main()