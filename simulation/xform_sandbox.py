# Copyright (c) 2025 Oguzhan Cagirir (OguzhanCOG), KCL Electronics Society
#
# Project: KCL FoAI RoboFootball System
# File: xform_sandbox.py
# Description: Utility module for transforming world-space AI targets into robot-local joystick commands and managing WebSocket communication.
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
# --- Version: 1.1.0 ---

import pygame
import math
import numpy as np 

import os
import time

WS_CLIENT_ENABLED = False
try:
    from ws_client import init_global_ws_client, get_global_ws_client, shutdown_global_ws_client
    WS_CLIENT_ENABLED = True
    print("SANDBOX_UTIL_INFO: ws_client module loaded. WebSocket functionality will be enabled.")
except ImportError:
    print("SANDBOX_UTIL_WARNING: ws_client.py not found. WebSocket functionality will be disabled.")
    _sandbox_printed_messages_dummy_ws = set()
    def _print_once_dummy_ws(key, msg):
        global _sandbox_printed_messages_dummy_ws
        if key not in _sandbox_printed_messages_dummy_ws: print(msg); _sandbox_printed_messages_dummy_ws.add(key)
    def init_global_ws_client(): _print_once_dummy_ws("ws_dummy_init", "SANDBOX_UTIL: (Dummy) init_global_ws_client called.")
    def get_global_ws_client(): return None 
    def shutdown_global_ws_client(): _print_once_dummy_ws("ws_dummy_shutdown", "SANDBOX_UTIL: (Dummy) shutdown_global_ws_client called.")

SANDBOX_MOVEMENT_SPEED_INCREMENT = 5.0 
WS_SEND_INTERVAL_SECONDS = 0.1       

g_ws_client_instance_util = None
g_last_ws_send_times_util = {} 

SANDBOX_BASE_ORIENTATION_VECTOR_UTIL = pygame.math.Vector2(1, 0) 

_sandbox_util_printed_messages = set()
def print_once_sb_util(message_key, message_content):
    global _sandbox_util_printed_messages
    if message_key not in _sandbox_util_printed_messages:
        print(message_content)
        _sandbox_util_printed_messages.add(message_key)

def initialize_sandbox_websocket_client():
    global g_ws_client_instance_util
    if not WS_CLIENT_ENABLED:
        print_once_sb_util("ws_init_disabled", "SANDBOX_UTIL_API: WebSocket client is disabled (import failed).")
        return False

    if g_ws_client_instance_util is not None and g_ws_client_instance_util.check_if_actively_connected():
        print_once_sb_util("ws_already_init", "SANDBOX_UTIL_API: WebSocket client already initialized and connected.")
        return True

    try:
        init_global_ws_client() 
        g_ws_client_instance_util = get_global_ws_client() 
        if g_ws_client_instance_util is None:
            print_once_sb_util("ws_init_fail_api", "SANDBOX_UTIL_API: WS client get_global_ws_client() returned None.")
            return False
        else:

            time.sleep(0.5) 
            if g_ws_client_instance_util.check_if_actively_connected():
                 print_once_sb_util("ws_init_ok", "SANDBOX_UTIL_API: WebSocket client initialized and connected.")
                 return True
            else:
                 print_once_sb_util("ws_init_not_conn", "SANDBOX_UTIL_API: WebSocket client initialized but not connected yet.")
                 return True 
    except Exception as e:
        print(f"SANDBOX_UTIL_API: Error initializing WebSocket client: {e}")
        g_ws_client_instance_util = None
        return False

def shutdown_sandbox_websocket_client():
    global g_ws_client_instance_util
    if not WS_CLIENT_ENABLED:
        print_once_sb_util("ws_shutdown_disabled", "SANDBOX_UTIL_API: WebSocket client was disabled. Nothing to shut down.")
        return

    print_once_sb_util("ws_shutdown_attempt", "SANDBOX_UTIL_API: Attempting to shut down WebSocket client...")
    try:
        shutdown_global_ws_client() 
        print_once_sb_util("ws_shutdown_ok", "SANDBOX_UTIL_API: WebSocket client shutdown command issued.")
    except Exception as e:
        print(f"SANDBOX_UTIL_API: Error during WebSocket client shutdown: {e}")
    g_ws_client_instance_util = None
    g_last_ws_send_times_util.clear()

def calculate_joystick_from_world_target(
        current_pos_m: tuple,
        ai_target_pos_m: tuple,
        current_orientation_deg: float, 
        movement_speed_increment: float = SANDBOX_MOVEMENT_SPEED_INCREMENT
    ) -> tuple: 
    target_vec_m = pygame.math.Vector2(ai_target_pos_m)
    current_pos_vec_m = pygame.math.Vector2(current_pos_m)
    desired_world_movement_m = target_vec_m - current_pos_vec_m

    desired_vel_sim_magnitude_input = pygame.math.Vector2(0, 0)
    if desired_world_movement_m.length_squared() > 1e-6:
        desired_vel_sim_magnitude_input = desired_world_movement_m.normalize() * movement_speed_increment

    current_robot_nose_vec = SANDBOX_BASE_ORIENTATION_VECTOR_UTIL.rotate(current_orientation_deg)
    current_robot_phys_right_vec = current_robot_nose_vec.rotate(90) 

    joy_y_local, joy_x_local = 0.0, 0.0
    if desired_vel_sim_magnitude_input.length_squared() > 1e-6:
        joy_y_unscaled = desired_vel_sim_magnitude_input.dot(current_robot_nose_vec)
        joy_x_unscaled = desired_vel_sim_magnitude_input.dot(current_robot_phys_right_vec)

        joy_y_local = joy_y_unscaled / movement_speed_increment if movement_speed_increment > 1e-6 else 0
        joy_x_local = joy_x_unscaled / movement_speed_increment if movement_speed_increment > 1e-6 else 0

        mag_joy = math.sqrt(joy_x_local**2 + joy_y_local**2)
        if mag_joy > 1.0:
            joy_x_local /= mag_joy
            joy_y_local /= mag_joy

        joy_x_local = max(-1.0, min(1.0, joy_x_local))
        joy_y_local = max(-1.0, min(1.0, joy_y_local))

    return joy_x_local, joy_y_local

def send_transformed_joystick_command_ws(
        websocket_user_id: int,
        joy_x: float,
        joy_y: float
    ):
    global g_ws_client_instance_util, g_last_ws_send_times_util

    current_time = time.time()
    last_send = g_last_ws_send_times_util.get(websocket_user_id, 0)

    if current_time - last_send >= WS_SEND_INTERVAL_SECONDS:
        if WS_CLIENT_ENABLED and g_ws_client_instance_util and g_ws_client_instance_util.check_if_actively_connected():
            try:
                g_ws_client_instance_util.send_joystick_command(userid=websocket_user_id, x=joy_x, y=joy_y)
                g_last_ws_send_times_util[websocket_user_id] = current_time

            except Exception as e:
                print_once_sb_util(f"ws_send_err_{websocket_user_id}", f"SANDBOX_UTIL_API: Error sending WS command for User {websocket_user_id}: {e}")

        elif WS_CLIENT_ENABLED and g_ws_client_instance_util:
            print_once_sb_util(f"ws_util_not_conn_send_{websocket_user_id}", f"SANDBOX_UTIL_API: WS client not connected. Cannot send for User {websocket_user_id}.")
        elif not WS_CLIENT_ENABLED:
            print_once_sb_util(f"ws_util_disabled_send_{websocket_user_id}", f"SANDBOX_UTIL_API: WS disabled. Faux send for User {websocket_user_id}: joy_x={joy_x:.2f}, joy_y={joy_y:.2f}")
            g_last_ws_send_times_util[websocket_user_id] = current_time 

def calculate_orientation_from_sim_corners(
        sim_corners_m: np.ndarray, 
        front_indices=(0,1), 
        back_indices=(3,2)   
    ) -> float | None: 
    if sim_corners_m is None or sim_corners_m.shape[0] != 4:
        return None

    c_world = sim_corners_m.reshape(4, 2) 

    p_back = (c_world[back_indices[0]] + c_world[back_indices[1]]) / 2.0
    p_front = (c_world[front_indices[0]] + c_world[front_indices[1]]) / 2.0

    vec_x = p_front[0] - p_back[0]
    vec_y = p_front[1] - p_back[1]

    if abs(vec_x) < 1e-6 and abs(vec_y) < 1e-6: 
        return None 

    raw_rad = math.atan2(vec_y, vec_x)
    raw_deg = math.degrees(raw_rad)

    orientation_deg = (raw_deg + 180) % 360 - 180
    return orientation_deg

# If __name__ == '__main__':
#    print("xform_sandbox.py is now primarily a utility module.")
#    print("You must re-work the standalone GUI to use integrated orientation or mock data.")
#    # initialize_sandbox_websocket_client()
#    # # Simulate some calls
#    # joy_x, joy_y = calculate_joystick_from_world_target((0,0), (1,0), 0)
#    # print(f"Simulated joy: {joy_x}, {joy_y}")
#    # if g_ws_client_instance_util: send_transformed_joystick_command_ws(99, joy_x, joy_y)
#    # time.sleep(2)
#    # shutdown_sandbox_websocket_client()