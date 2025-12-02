# Copyright (c) 2025 Oguzhan Cagirir (OguzhanCOG), KCL Electronics Society
#
# Project: KCL FoAI RoboFootball System
# File: main.py
# Description: Main RoboFootball simulation environment, including game logic, AI behaviour, and AprilTag-based perception.
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
# --- Development Powered by NeuralFusion(TM) III ---
# --- Version: 1.7.1 ---

import pygame
import math
import random
import traceback
import cv2
import pupil_apriltags
import numpy as np
import os
import threading
import queue
import time
import xform_sandbox as sandbox_utils

_PLAYER_RADIUS_M_VAL = 0.065
_BALL_RADIUS_M_VAL = 0.025
_ARENA_WIDTH_M_VAL = 2.4
_ARENA_HEIGHT_M_VAL = 2.0
_GOAL_WIDTH_M_VAL = 0.4

CFG = {
    "SCREEN_WIDTH": 1536,
    "SCREEN_HEIGHT": 864,
    "FPS": 60,
    "MARGIN": 50,
    "ARENA_WIDTH_M": _ARENA_WIDTH_M_VAL,
    "ARENA_HEIGHT_M": _ARENA_HEIGHT_M_VAL,
    "PLAYER_RADIUS_M": _PLAYER_RADIUS_M_VAL,
    "BALL_RADIUS_M": _BALL_RADIUS_M_VAL,
    "GOAL_WIDTH_M": _GOAL_WIDTH_M_VAL,
    "GOAL_DEPTH_M": 0.1,
    "PLAYER_MASS": 10.0,
    "BALL_MASS": 1.0,
    "PLAYER_ACCELERATION": 15.0, 
    "PLAYER_MAX_SPEED": 1.8,    
    "PLAYER_DRAG": 4.0,
    "BALL_DAMPING": 0.9,
    "RESTITUTION_PLAYER_BALL": 0.6,
    "RESTITUTION_PLAYER_PLAYER": 0.4,
    "RESTITUTION_BALL_WALL": 0.7,

    "AI_MAX_SPEED": 1.7,
    "AI_ACCELERATION": 12.0,
    "AI_MAX_FORCE": 170.0,
    "AI_Slowing_RADIUS_BALL": 0.45,
    "AI_Slowing_RADIUS_POSITION": 0.3,
    "AI_SEPARATION_DISTANCE": _PLAYER_RADIUS_M_VAL * 7.0,
    "AI_ARRIVAL_THRESHOLD_FACTOR": 0.25,
    "AI_POSITIONING_TARGET_UPDATE_THRESHOLD_FACTOR": 0.5,
    "AI_AT_BALL_FOR_DECISION_M": _PLAYER_RADIUS_M_VAL + _BALL_RADIUS_M_VAL + 0.025,
    "AI_HAS_BALL_THRESHOLD_M": _PLAYER_RADIUS_M_VAL + _BALL_RADIUS_M_VAL + 0.05,
    "AI_CAN_KICK_THRESHOLD_M": _PLAYER_RADIUS_M_VAL + _BALL_RADIUS_M_VAL + 0.065,
    "AI_KICK_COOLDOWN_FRAMES": 10,
    "AI_SHOOTING_MAX_RANGE_M": _ARENA_WIDTH_M_VAL * 0.50,
    "AI_SHOOTING_CLEAR_PATH_WIDTH_M": _PLAYER_RADIUS_M_VAL * 1.8,
    "AI_SHOOT_MAX_X_OWN_HALF_B": 0.65,
    "AI_SHOOT_MIN_X_OWN_HALF_A": 0.35,
    "AI_SHOOT_FACING_ANGLE_DEGREES": 60.0,
    "AI_DRIBBLE_DISTANCE_FACTOR": 5.0,
    "AI_DRIBBLE_PERSIST_FRAMES": 8,
    "AI_DRIBBLE_PATH_EXTRA_CLEARANCE_M": 0.03,
    "AI_DRIBBLE_GOAL_DIRECTION_BONUS": 0.3,
    "AI_PASS_MAX_DIST_M": _ARENA_WIDTH_M_VAL * 0.6,
    "AI_PASS_MIN_DIST_M": _PLAYER_RADIUS_M_VAL * 5.0,
    "AI_PASS_OPENNESS_CHECK_WIDTH_M": _PLAYER_RADIUS_M_VAL * 2.2,
    "AI_PASS_TEAMMATE_FORWARD_ADVANTAGE_M": _PLAYER_RADIUS_M_VAL * 3.0,
    "AI_CLEAR_PASS_TEAMMATE_MARKED_RADIUS_FACTOR": 4.5,
    "AI_CLEAR_PASS_MIN_DIST_M": _PLAYER_RADIUS_M_VAL * 4.0,
    "AI_CLEAR_PASS_MAX_DIST_M": _ARENA_WIDTH_M_VAL * 0.35,
    "AI_HOOF_CLEARANCE_WIDTH_M": _PLAYER_RADIUS_M_VAL * 1.5,
    "AI_DEFENSIVE_THIRD_LINE_X_FACTOR_B": 0.70,
    "AI_DEFENSIVE_THIRD_LINE_X_FACTOR_A": 0.30,
    "AI_THREAT_DISTANCE_TO_GOAL_M": _ARENA_WIDTH_M_VAL * 0.40,
    "AI_DEFENDER_INTERCEPT_STANDOFF_M": _ARENA_WIDTH_M_VAL * 0.20,
    "AI_DEFENDER_COVER_SPACE_OFFSET_M": _PLAYER_RADIUS_M_VAL * 5.0,
    "AI_DEFENDER_MIN_X_B_FACTOR": 0.45,
    "AI_DEFENDER_MAX_X_A_FACTOR": 0.55,
    "AI_DEF_COVER_PERSIST_FRAMES": 6,
    "AI_DEF_COVER_THREAT_CONE_DOT": 0.1,
    "AI_MIDFIELD_HOLD_BALL_Y_DEADZONE_M": 0.25,
    "AI_BALL_WINNER_PROXIMITY_ADVANTAGE_M": 0.22,
    "AI_ATTACKER_OFFBALL_X_ZONE_MIN_FACTOR": 0.10,
    "AI_ATTACKER_OFFBALL_X_ZONE_MAX_FACTOR": 0.48,
    "AI_ATTACKER_OFFBALL_Y_SPREAD_FACTOR": 0.45,
    "AI_ATTACKER_SUPPORT_BEHIND_BALL_OFFSET_M": _PLAYER_RADIUS_M_VAL * 3.5,
    "AI_DEFENDER_MIDFIELD_HOLD_X_FACTOR_B": 0.60,
    "AI_DEFENDER_MIDFIELD_HOLD_X_FACTOR_A": 0.40,
    "AI_WEIGHT_PRIMARY_OBJECTIVE": 1.0,
    "AI_WEIGHT_SEPARATION": 1.75,

    "COLOR_BACKGROUND": (30,30,30), "COLOR_PITCH": (0,120,0), "COLOR_LINES": (200,200,200),
    "COLOR_TEAM_A": (220,50,50), "COLOR_TEAM_B": (50,100,220), "COLOR_BALL": (255,200,0),
    "COLOR_GOAL": (230,230,230),

    "MAX_SCORE": 5, "RESET_DELAY_MS": 3000,

    "DEBUG_LOG_AI_GENERAL": True, "DEBUG_LOG_AI_BT_TICKS": False, "DEBUG_LOG_AI_CONDITIONS": False,
    "DEBUG_AI_PLAYER_FOCUS_NUM": 0, "DEBUG_AI_VELOCITY_VECTORS": True,
    "DEBUG_LOG_GAME_EVENTS": True, "DEBUG_LOG_AI_PATHING": False, "DEBUG_LOG_AI_DRIBBLE_CHOICE": False,
    "DEBUG_LOG_AI_CLEAR_STRATEGY": False, "DEBUG_LOG_AI_STEERING": False,
}

CALIBRATION_DATA_FOLDER = "camera_calibration_data"
CAMERA_CALIBRATION_FILE_TEMPLATE = "camera_calibration_{width}x{height}.npz"
PERSPECTIVE_MATRICES_NPZ_FILE = "perspective_transform_matrices.npz"

APRILTAG_FAMILY = "tag36h11"
TAG_SIZE_METERS = 0.093 
CAMERA_INDEX = 1 
APRILTAG_CAMERA_RESOLUTION_W = 1280
APRILTAG_CAMERA_RESOLUTION_H = 960

ROBOT_B1_TAG_ID = 4
ROBOT_B2_TAG_ID = 5
ROBOT_A1_TAG_ID = 6
ROBOT_A2_TAG_ID = 7
ALL_ROBOT_TAG_IDS = [ROBOT_B1_TAG_ID, ROBOT_B2_TAG_ID, ROBOT_A1_TAG_ID, ROBOT_A2_TAG_ID] 

LOWER_YELLOW_HSV = np.array([20, 100, 100])
UPPER_YELLOW_HSV = np.array([35, 255, 255])
BALL_MORPH_KERNEL_SIZE = 5
BALL_ERODE_ITERATIONS = 1
BALL_DILATE_ITERATIONS = 1
MIN_BALL_CONTOUR_AREA_PX = 50
MIN_BALL_CIRCULARITY = 0.7

DETECTOR_KWARGS = {
    'families': APRILTAG_FAMILY, 'nthreads': 1, 'quad_decimate': 1.0,
    'quad_sigma': 0.0, 'refine_edges': True, 'decode_sharpening': 0.25, 'debug': False
}

_AI_LOG_FILE_PATH = "main_sim_log.txt"
_ai_log_file_handle = None

METERS_TO_PIXELS = 1.0
PITCH_RECT_PX = pygame.Rect(0,0,0,0)
GOAL_LINE_THICKNESS_PX = 6
LINE_THICKNESS_PX = 3

apriltag_thread = None
apriltag_queue = queue.Queue(maxsize=10) 
stop_apriltag_thread = threading.Event()
camera_matrix_at, dist_coeffs_at = None, None
perspective_matrix_metric_to_sim_at = None
perspective_matrix_pixel_to_sim_at = None 
apriltag_detector_at, camera_capture_at = None, None
_sim_printed_messages = set()

def print_once(message_key, message_content):
    global _sim_printed_messages
    if message_key not in _sim_printed_messages:
        print(message_content)
        _sim_printed_messages.add(message_key)

def log_ai(player_obj, message_type, message):
    global _ai_log_file_handle
    if not CFG.get("DEBUG_LOG_AI_GENERAL", False): return

    player_focus_num = CFG.get("DEBUG_AI_PLAYER_FOCUS_NUM", 0)
    is_bt_driven = isinstance(player_obj, Player) and getattr(player_obj, 'is_ai_driven_by_bt', False)

    if player_obj and player_focus_num != 0:

        is_focused = (hasattr(player_obj, 'player_num') and
                      player_obj.player_num == player_focus_num and
                      hasattr(player_obj, 'team') and player_obj.team == 'B')
        if not is_focused and message_type != "GAME_LOG":
            return

        if is_focused and not is_bt_driven and message_type not in ["GAME_LOG", "EVENT"]: 
            return

    if message_type == "BT_TICK" and not CFG.get("DEBUG_LOG_AI_BT_TICKS", False): return
    if message_type == "CONDITION" and not CFG.get("DEBUG_LOG_AI_CONDITIONS", False): return
    if message_type == "PATH_FAIL" and not CFG.get("DEBUG_LOG_AI_PATHING", False): return
    if message_type == "DRIBBLE_CHOICE" and not CFG.get("DEBUG_LOG_AI_DRIBBLE_CHOICE", False): return
    if message_type == "CLEAR_STRATEGY" and not CFG.get("DEBUG_LOG_AI_CLEAR_STRATEGY", False): return
    if message_type == "STEERING_FORCES" and not CFG.get("DEBUG_LOG_AI_STEERING", False): return

    log_source = f" ({player_obj})" if player_obj else ""
    log_line = f"LOG{log_source}: [{message_type}] {message}\n"

    if _ai_log_file_handle and not _ai_log_file_handle.closed:
        try:
            _ai_log_file_handle.write(log_line)
            _ai_log_file_handle.flush()
        except Exception as e:
            print(f"ERROR WRITING TO LOG FILE: {e}")
            print(f"FALLBACK CONSOLE LOG: {log_line.strip()}") 
    else:

        print(f"NO AI LOG FILE (or closed): {log_line.strip()}")

def safe_normalize(vector):
    len_sq = vector.length_squared()
    if len_sq < 1e-9:
        return pygame.math.Vector2(0, 0)

    return vector / math.sqrt(len_sq)

class Entity: 
    def __init__(self, pos_m, radius_m, mass, color, max_speed_m_s=0, drag=0):
        self.pos_m = pygame.math.Vector2(pos_m)
        self.vel_m_s = pygame.math.Vector2(0, 0)
        self.acc_m_s2 = pygame.math.Vector2(0, 0)
        self.radius_m = radius_m
        self.mass = mass if mass > 1e-6 else 1e-6 
        self.color = color
        self.max_speed_m_s = max_speed_m_s
        self.drag = drag

    def get_screen_pos(self):
        return pygame.math.Vector2(
            PITCH_RECT_PX.x + self.pos_m.x * METERS_TO_PIXELS,
            PITCH_RECT_PX.y + self.pos_m.y * METERS_TO_PIXELS)

    def get_screen_radius(self):
        return self.radius_m * METERS_TO_PIXELS

    def update(self, dt): 
        if self.acc_m_s2.length_squared() > 0:
             self.vel_m_s += self.acc_m_s2 * dt
        elif self.drag > 0:
             self.vel_m_s *= (1 - self.drag * dt)
             if self.vel_m_s.length_squared() < (0.01 * 0.01): 
                 self.vel_m_s.update(0,0)

        if self.max_speed_m_s > 0 and self.vel_m_s.length_squared() > self.max_speed_m_s**2:
            self.vel_m_s.scale_to_length(self.max_speed_m_s)

        if not (isinstance(self, Player) and hasattr(self, 'tag_id_link') and self.tag_id_link is not None):
            self.pos_m += self.vel_m_s * dt

        self.acc_m_s2.update(0, 0) 

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.get_screen_pos(), self.get_screen_radius())

        is_ai_player = isinstance(self, Player) and getattr(self, 'is_ai_driven_by_bt', False)
        if CFG["DEBUG_AI_VELOCITY_VECTORS"] and self.vel_m_s.length_squared() > 0.01 and is_ai_player:
             start_pos_px = self.get_screen_pos()
             end_pos_px = start_pos_px + self.vel_m_s * METERS_TO_PIXELS * 0.3 
             pygame.draw.line(screen, (255,255,255), start_pos_px, end_pos_px, 1)

class BTNode:
    def __init__(self, name="Node", player=None):
        self.name = name
        self.player_context = player
        self._status = "FAILURE"
    def get_status(self): return self._status
    def set_status(self, status): self._status = status
    def reset(self):
        self.set_status("FAILURE")
        if hasattr(self, 'children'): 
            for child in self.children:
                if hasattr(child, 'reset'): 
                    child.reset()
    def tick(self, game_state): raise NotImplementedError
    def __str__(self): return f"{self.__class__.__name__}({self.name}, St: {self._status})"

class CompositeNode(BTNode):
    def __init__(self, name, children=None, player=None): 
        super().__init__(name=name, player=player)       
        self.children = children if children is not None else []
        for child in self.children:
            if child.player_context is None:
                child.player_context = player
        self.running_child_idx = -1
    def add_child(self, child_node):
        if child_node.player_context is None:
            child_node.player_context = self.player_context
        self.children.append(child_node)
    def reset(self):
        super().reset()
        self.running_child_idx = -1

class Selector(CompositeNode):
    def __init__(self, name, children=None, player=None): 
        super().__init__(name=name, children=children, player=player) 

    def tick(self, game_state):
        start_idx = 0
        if self.running_child_idx != -1: start_idx = self.running_child_idx
        for i in range(start_idx, len(self.children)):
            child = self.children[i]
            if self.running_child_idx != i: child.reset()
            child_status = child.tick(game_state)
            if child_status == "SUCCESS" or child_status == "RUNNING":
                self.set_status(child_status)
                self.running_child_idx = i if child_status == "RUNNING" else -1
                return self.get_status()
        self.set_status("FAILURE"); self.running_child_idx = -1; return self.get_status()

class Sequence(CompositeNode):
    def __init__(self, name, children=None, player=None): 
        super().__init__(name=name, children=children, player=player) 

    def tick(self, game_state):
        start_idx = 0
        if self.running_child_idx != -1: start_idx = self.running_child_idx
        for i in range(start_idx, len(self.children)):
            child = self.children[i]
            if self.running_child_idx != i: child.reset()
            child_status = child.tick(game_state)
            if child_status == "FAILURE" or child_status == "RUNNING":
                self.set_status(child_status)
                self.running_child_idx = i if child_status == "RUNNING" else -1
                return self.get_status()
        self.set_status("SUCCESS"); self.running_child_idx = -1; return self.get_status()

class ActionNode(BTNode):
    def __init__(self, action_name, player=None):
        super().__init__(name=f"Act_{action_name}", player=player)
        self.action_true_name = action_name
    def tick(self, game_state):
        log_ai(self.player_context, "BT_TICK", f"Ticking Action: {self.name} (Current status: {self._status})")
        action_method = getattr(self.player_context, f"action_{self.action_true_name}", None)
        if action_method:
            status = action_method(game_state); self.set_status(status)
            log_ai(self.player_context, "BT_TICK", f"Action {self.name} new status: {status}")
        else:
            log_ai(self.player_context, "ERROR", f"Action method 'action_{self.action_true_name}' not found for {self.player_context}!")
            self.set_status("FAILURE")
        return self.get_status()

class ConditionNode(BTNode):
    def __init__(self, condition_name, player=None, negate=False):
        super().__init__(name=f"Cond_{condition_name}{'_Not' if negate else ''}", player=player)
        self.condition_true_name = condition_name; self.negate = negate
    def tick(self, game_state):
        condition_method = getattr(self.player_context, f"condition_{self.condition_true_name}", None)
        if condition_method:
            result = condition_method(game_state); actual_result = not result if self.negate else result
            self.set_status("SUCCESS" if actual_result else "FAILURE")
            log_ai(self.player_context, "CONDITION", f"{self.name} eval: {result} (negated: {self.negate}) -> {self.get_status()}")
        else:
            log_ai(self.player_context, "ERROR", f"Condition method 'condition_{self.condition_true_name}' not found for {self.player_context}!")
            self.set_status("FAILURE")
        return self.get_status()

class Player(Entity): 
    def __init__(self, pos_m, team_char, player_num_role, is_ai_driven_by_bt=False):
        self.team = team_char
        self.player_num = player_num_role 
        self.is_ai_driven_by_bt = is_ai_driven_by_bt
        self.tag_id_link = None 

        cfg_prefix = "AI_" if self.is_ai_driven_by_bt else "PLAYER_" 
        max_speed = CFG[f"{cfg_prefix}MAX_SPEED"]
        self.acceleration_magnitude = CFG[f"{cfg_prefix}ACCELERATION"]
        drag = CFG["PLAYER_DRAG"] 
        color = CFG["COLOR_TEAM_A"] if self.team == 'A' else CFG["COLOR_TEAM_B"]

        super().__init__(pos_m, CFG["PLAYER_RADIUS_M"], CFG["PLAYER_MASS"], color, max_speed, drag)

        self.primary_steering_target_m = None
        self.slowing_radius_for_primary = CFG["AI_Slowing_RADIUS_BALL"] 
        self.current_action_name_for_debug = "Idle_Init"
        self.behavior_tree = None
        self._last_has_ball_status = False
        self._last_designated_winner_status = False 
        self.kick_cooldown_timer = 0

        self.dribble_target_m = None
        self.dribble_target_persists_frames = 0
        self.last_ball_y_for_midfield_hold = -1.0
        self.defensive_cover_target_m = None
        self.defensive_cover_persists_frames = 0

        if self.is_ai_driven_by_bt:
            self.max_force = CFG["AI_MAX_FORCE"] 
            self._build_behavior_tree() 
            if self.behavior_tree:
                self.behavior_tree.reset()
        else: 
            self.current_action_name_for_debug = f"Human_A{self.player_num}"
            self.max_force = 0 

    def __str__(self): 
        role_suffix = ""
        if self.is_ai_driven_by_bt:
            role_suffix = "_AI_B" + str(self.player_num) + ("(Att)" if self.player_num == 1 else "(Def)")
        else: 
            role_suffix = f"_Human_A{self.player_num}"
        return f"Player_{self.team}{self.player_num}{role_suffix}"

    def _arrive(self, target_pos_m, slowing_radius): 
        desired_offset = target_pos_m - self.pos_m
        distance = desired_offset.length()
        if distance > 0.001: 
            ramped_speed = self.max_speed_m_s
            if distance < slowing_radius and slowing_radius > 1e-5: 
                ramped_speed = self.max_speed_m_s * (distance / slowing_radius)
            desired_velocity = (desired_offset / distance) * ramped_speed
            return desired_velocity - self.vel_m_s
        return pygame.math.Vector2(0, 0)

    def _separation(self, teammates, separation_distance): 
        avg_away_dir = pygame.math.Vector2(0, 0)
        count = 0
        for mate in teammates:
            dist_to_mate = self.pos_m.distance_to(mate.pos_m)
            if 1e-5 < dist_to_mate < separation_distance: 
                direction = safe_normalize(self.pos_m - mate.pos_m)
                avg_away_dir += direction / (dist_to_mate + 1e-5) 
                count += 1
        if count > 0:
            avg_away_dir /= count 
            if avg_away_dir.length_squared() > 0:
                desired_sep_velocity = avg_away_dir.normalize() * self.max_speed_m_s
                return desired_sep_velocity - self.vel_m_s
        return pygame.math.Vector2(0, 0)

    def _apply_steering_force(self, steering_force): 
        if steering_force.length_squared() > self.max_force**2:
            steering_force.scale_to_length(self.max_force)
        if steering_force.length_squared() > 0:
            self.acc_m_s2 = safe_normalize(steering_force) * self.acceleration_magnitude
        else:
            self.acc_m_s2.update(0,0) 

    def set_movement_input(self, direction_vector): 
        pass 

    def _get_game_context_objects(self, game_state): 
        ball = game_state['ball']
        teammates = game_state['teammates']
        opponents = game_state['opponents']
        is_designated_ball_winner = game_state.get('is_designated_ball_winner', False)
        my_goal_center = pygame.math.Vector2(CFG["ARENA_WIDTH_M"], CFG["ARENA_HEIGHT_M"] / 2) if self.team == 'B' else pygame.math.Vector2(0, CFG["ARENA_HEIGHT_M"] / 2)
        opponent_goal_center = pygame.math.Vector2(0, CFG["ARENA_HEIGHT_M"] / 2) if self.team == 'B' else pygame.math.Vector2(CFG["ARENA_WIDTH_M"], CFG["ARENA_HEIGHT_M"] / 2)
        return ball, teammates, opponents, my_goal_center, opponent_goal_center, is_designated_ball_winner

    def condition_HasBall(self, game_state): ball,_,_,_,_,_ = self._get_game_context_objects(game_state); return self.pos_m.distance_squared_to(ball.pos_m) < CFG["AI_HAS_BALL_THRESHOLD_M"]**2
    def condition_CanKickBall(self, game_state):
        if self.kick_cooldown_timer > 0: return False
        ball,_,_,_,_,_ = self._get_game_context_objects(game_state); return self.pos_m.distance_squared_to(ball.pos_m) < CFG["AI_CAN_KICK_THRESHOLD_M"]**2
    def condition_TeammateHasBall(self, game_state): ball,teammates,_,_,_,_ = self._get_game_context_objects(game_state); return any(t.pos_m.distance_squared_to(ball.pos_m) < CFG["AI_HAS_BALL_THRESHOLD_M"]**2 for t in teammates)
    def condition_BallInMyDefensiveThird(self, game_state): ball,_,_,_,_,_ = self._get_game_context_objects(game_state); line = CFG["ARENA_WIDTH_M"]*(CFG["AI_DEFENSIVE_THIRD_LINE_X_FACTOR_B"] if self.team=='B' else CFG["AI_DEFENSIVE_THIRD_LINE_X_FACTOR_A"]); return ball.pos_m.x > line if self.team=='B' else ball.pos_m.x < line
    def condition_BallInOpponentHalf(self, game_state): ball,_,_,_,_,_ = self._get_game_context_objects(game_state); return ball.pos_m.x < CFG["ARENA_WIDTH_M"]/2 if self.team=='B' else ball.pos_m.x > CFG["ARENA_WIDTH_M"]/2
    def _is_path_clear(self, start_pos, end_pos, obstacles, clearance_radius): 
        path_vec = end_pos - start_pos; path_len_sq = path_vec.length_squared()
        if path_len_sq < 1e-6: return True
        path_dir = safe_normalize(path_vec); path_len = math.sqrt(path_len_sq)
        for obs in obstacles:
            if obs is self: continue
            to_obs_vec = obs.pos_m - start_pos; proj_scalar = to_obs_vec.dot(path_dir)
            dist_to_line_sq = 0
            if proj_scalar < 0: dist_to_line_sq = to_obs_vec.length_squared()
            elif proj_scalar > path_len: dist_to_line_sq = (obs.pos_m - end_pos).length_squared()
            else: perpendicular_vec = to_obs_vec - (path_dir * proj_scalar); dist_to_line_sq = perpendicular_vec.length_squared()
            combined_radius_for_collision = obs.radius_m + clearance_radius
            if dist_to_line_sq < combined_radius_for_collision**2:
                if -obs.radius_m < proj_scalar < path_len + obs.radius_m:
                    if CFG.get("DEBUG_LOG_AI_PATHING", False): log_ai(self, "PATH_FAIL", f"Path blocked by {obs} for {self.current_action_name_for_debug}")
                    return False
        return True
    def condition_CanShoot(self, game_state): 
        ball, _, opponents, my_goal_center, opponent_goal_center, _ = self._get_game_context_objects(game_state)
        if not self.condition_CanKickBall(game_state): return False
        player_focus_check = (CFG.get("DEBUG_AI_PLAYER_FOCUS_NUM", 0) in [0, self.player_num]) if self.is_ai_driven_by_bt else False 
        own_half_shoot_limit_factor_b = CFG.get("AI_SHOOT_MAX_X_OWN_HALF_B", 0.65)
        if self.team == 'B' and self.pos_m.x > CFG["ARENA_WIDTH_M"] * own_half_shoot_limit_factor_b:
            if CFG.get("DEBUG_LOG_AI_CONDITIONS", False) and player_focus_check: log_ai(self, "CONDITION", "CanShoot: False (B player too deep)")
            return False
        dist_to_goal_sq = self.pos_m.distance_squared_to(opponent_goal_center)
        if dist_to_goal_sq > CFG["AI_SHOOTING_MAX_RANGE_M"]**2:
            if CFG.get("DEBUG_LOG_AI_CONDITIONS", False) and player_focus_check: log_ai(self, "CONDITION", "CanShoot: False (out of range)")
            return False
        shoot_facing_angle_rad = math.radians(CFG.get("AI_SHOOT_FACING_ANGLE_DEGREES", 60.0))
        if self.vel_m_s.length_squared() > 0.01: 
            dir_to_goal = safe_normalize(opponent_goal_center - self.pos_m)
            if dir_to_goal.length_squared() > 1e-6: 
                facing_dir = safe_normalize(self.vel_m_s)
                dot_product = max(-1.0, min(1.0, dir_to_goal.dot(facing_dir))) 
                if dot_product < math.cos(shoot_facing_angle_rad):
                    if CFG.get("DEBUG_LOG_AI_CONDITIONS", False) and player_focus_check: log_ai(self, "CONDITION", "CanShoot: False (not facing goal well)")
                    return False
        all_other_players = [p for p in opponents + game_state.get('teammates', []) if p is not self]
        path_is_clear = self._is_path_clear(self.pos_m, opponent_goal_center, all_other_players, CFG["BALL_RADIUS_M"] + CFG["AI_SHOOTING_CLEAR_PATH_WIDTH_M"]/2)
        if not path_is_clear and CFG.get("DEBUG_LOG_AI_CONDITIONS", False) and player_focus_check: log_ai(self, "CONDITION", "CanShoot: False (path not clear)")
        return path_is_clear
    def condition_CanPassToTeammate(self, game_state): 
        ball, teammates, opponents, _, _, _ = self._get_game_context_objects(game_state)
        if not teammates or not self.condition_CanKickBall(game_state): return False
        teammate = teammates[0] 
        dist_to_teammate = self.pos_m.distance_to(teammate.pos_m)
        if not (CFG["AI_PASS_MIN_DIST_M"] < dist_to_teammate < CFG["AI_PASS_MAX_DIST_M"]): return False
        adv_check = CFG["AI_PASS_TEAMMATE_FORWARD_ADVANTAGE_M"]
        is_teammate_forward = (teammate.pos_m.x < self.pos_m.x - adv_check if self.team == 'B' else teammate.pos_m.x > self.pos_m.x + adv_check)
        if not is_teammate_forward: return False
        return self._is_path_clear(self.pos_m, teammate.pos_m, opponents, ball.radius_m + CFG["AI_PASS_OPENNESS_CHECK_WIDTH_M"]/2)
    def condition_IsDesignatedBallWinner(self, game_state): _,_,_,_,_,is_winner = self._get_game_context_objects(game_state); return is_winner
    def condition_IsMyGoalThreatened(self, game_state): 
        ball, _, opponents, my_goal_center, _, _ = self._get_game_context_objects(game_state)
        is_ball_advancing = (ball.vel_m_s.x > 0.35 if self.team == 'B' else ball.vel_m_s.x < -0.35)
        ball_in_def_third = self.condition_BallInMyDefensiveThird(game_state)
        close_to_goal_sq = ball.pos_m.distance_squared_to(my_goal_center) < CFG["AI_THREAT_DISTANCE_TO_GOAL_M"]**2
        for opp in opponents:
            if opp.pos_m.distance_squared_to(ball.pos_m) < (CFG["AI_HAS_BALL_THRESHOLD_M"] * 1.3)**2:
                if opp.pos_m.distance_squared_to(my_goal_center) < (CFG["AI_THREAT_DISTANCE_TO_GOAL_M"] * 1.2)**2: return True
        return (close_to_goal_sq and (ball_in_def_third or is_ball_advancing))

    def _set_steering_target_for_action(self, target_pos, slowing_radius_key, action_name): 
        new_target = pygame.math.Vector2(target_pos); log_detail = False
        if self.current_action_name_for_debug != action_name: log_detail = True
        elif self.primary_steering_target_m and self.primary_steering_target_m.distance_squared_to(new_target) > (self.radius_m * 0.75)**2: log_detail = True
        self.primary_steering_target_m = new_target; self.slowing_radius_for_primary = CFG[slowing_radius_key]
        if log_detail: log_ai(self, "ACTION_INIT", f"Start/Upd: {action_name}, Tgt:({target_pos.x:.2f},{target_pos.y:.2f})")
        self.current_action_name_for_debug = action_name
    def _is_at_target(self, target_threshold_factor=None): 
        factor = target_threshold_factor or CFG["AI_ARRIVAL_THRESHOLD_FACTOR"]
        if not self.primary_steering_target_m: return False
        return self.pos_m.distance_squared_to(self.primary_steering_target_m) < (self.radius_m * factor)**2
    def action_GoToBall(self, game_state): ball,_,_,_,_,_ = self._get_game_context_objects(game_state); self._set_steering_target_for_action(ball.pos_m, "AI_Slowing_RADIUS_BALL", "GoToBall"); return "SUCCESS" if self.pos_m.distance_squared_to(ball.pos_m) < CFG["AI_AT_BALL_FOR_DECISION_M"]**2 else "RUNNING"
    def _execute_kick_action(self, target_pos, action_name, game_state): 
        if not self.condition_CanKickBall(game_state): return "FAILURE"
        self.kick_cooldown_timer = CFG["AI_KICK_COOLDOWN_FRAMES"]
        log_ai(self, "ACTION_EVENT", f"{action_name}: KICKED. Target ~({target_pos.x:.2f},{target_pos.y:.2f})")

        if self.behavior_tree: self.behavior_tree.reset() 
        return "SUCCESS"
    def action_ShootAtGoal(self, game_state): _,_,_,_,opponent_goal_center,_ = self._get_game_context_objects(game_state); return self._execute_kick_action(opponent_goal_center,"ShootAtGoal",game_state)
    def action_PassToTeammate(self, game_state): 
        _,teammates,_,_,_,_ = self._get_game_context_objects(game_state)
        if not teammates: return "FAILURE"
        return self._execute_kick_action(teammates[0].pos_m,"PassToTeammate",game_state)
    def action_ClearBall(self, game_state): 
        if not self.condition_CanKickBall(game_state): return "FAILURE"
        ball, teammates, opponents, _, opponent_goal_center, _ = self._get_game_context_objects(game_state)
        best_clear_target, clear_action_name = None, "ClearBall_DefaultWide"
        if teammates: 
            teammate = teammates[0] 
            adv_check = CFG.get("AI_PASS_TEAMMATE_FORWARD_ADVANTAGE_M", self.radius_m*3)*0.7
            is_fwd = (teammate.pos_m.x < self.pos_m.x-adv_check if self.team=='B' else teammate.pos_m.x > self.pos_m.x+adv_check)
            if is_fwd and self._is_path_clear(self.pos_m, teammate.pos_m, opponents, ball.radius_m + CFG["AI_PASS_OPENNESS_CHECK_WIDTH_M"]/2):
                is_open=True; marked_r_sq=(self.radius_m*CFG.get("AI_CLEAR_PASS_TEAMMATE_MARKED_RADIUS_FACTOR",4.5))**2
                for opp in opponents:
                    if opp.pos_m.distance_squared_to(teammate.pos_m) < marked_r_sq: is_open=False; break
                if is_open:
                    pass_dist_sq = self.pos_m.distance_squared_to(teammate.pos_m)
                    min_d, max_d = CFG.get("AI_CLEAR_PASS_MIN_DIST_M",self.radius_m*4), CFG.get("AI_CLEAR_PASS_MAX_DIST_M",CFG["ARENA_WIDTH_M"]*0.4)
                    if min_d**2 < pass_dist_sq < max_d**2: best_clear_target,clear_action_name = teammate.pos_m, "ClearBall_PassTeam"
        if best_clear_target is None: 
            tx, y_off_base = opponent_goal_center.x, CFG["GOAL_WIDTH_M"]*0.8+CFG["ARENA_HEIGHT_M"]*0.2
            y_off = y_off_base * random.uniform(0.9,1.3)
            cy1, cy2 = CFG["ARENA_HEIGHT_M"]/2+y_off, CFG["ARENA_HEIGHT_M"]/2-y_off
            pref_y = cy1 if self.pos_m.y < CFG["ARENA_HEIGHT_M"]/2 else cy2
            alt_y = cy2 if self.pos_m.y < CFG["ARENA_HEIGHT_M"]/2 else cy1
            pref_t, alt_t = pygame.math.Vector2(tx,pref_y), pygame.math.Vector2(tx,alt_y)
            pref_t.y=max(ball.radius_m*1.5,min(pref_t.y,CFG["ARENA_HEIGHT_M"]-ball.radius_m*1.5))
            alt_t.y=max(ball.radius_m*1.5,min(alt_t.y,CFG["ARENA_HEIGHT_M"]-ball.radius_m*1.5))
            hoof_clr = CFG.get("AI_HOOF_CLEARANCE_WIDTH_M", self.radius_m*1.5)
            path_pref = self._is_path_clear(self.pos_m,pref_t,opponents,ball.radius_m+hoof_clr/2)
            path_alt = self._is_path_clear(self.pos_m,alt_t,opponents,ball.radius_m+hoof_clr/2)
            if path_pref and path_alt: best_clear_target=pref_t if abs(pref_t.y-self.pos_m.y)>abs(alt_t.y-self.pos_m.y) else alt_t; clear_action_name="ClearBall_WideChosen"
            elif path_pref: best_clear_target=pref_t; clear_action_name="ClearBall_WidePref"
            elif path_alt: best_clear_target=alt_t; clear_action_name="ClearBall_WideAlt"
            else: best_clear_target=opponent_goal_center; clear_action_name="ClearBall_Desperate"
        return self._execute_kick_action(best_clear_target, clear_action_name, game_state)
    def action_DribbleStrategically(self, game_state): 
        if not self.condition_HasBall(game_state): self.dribble_target_m=None;self.dribble_target_persists_frames=0;return "FAILURE"
        ball,_,opps,my_goal,opp_goal,_ = self._get_game_context_objects(game_state)
        recalc = self.dribble_target_m is None or self.dribble_target_persists_frames<=0 or \
                 (self.primary_steering_target_m and self.pos_m.distance_squared_to(self.primary_steering_target_m)<(self.radius_m*CFG["AI_ARRIVAL_THRESHOLD_FACTOR"]*0.7)**2)
        act_name = "DribbleStrategically"
        if recalc:
            best_tgt, max_open = None, -float('inf')
            drib_dist = self.radius_m*CFG.get("AI_DRIBBLE_DISTANCE_FACTOR",5.0)
            for angle_off in [0,-25,25,-50,50,-75,75]: 
                dir_goal_base = safe_normalize(opp_goal-self.pos_m)
                drib_dir = dir_goal_base.rotate(angle_off) if dir_goal_base.length_squared()>1e-6 else pygame.math.Vector2(random.uniform(-1,1),random.uniform(-1,1)).normalize()
                pot_tgt = self.pos_m + drib_dir*drib_dist
                pot_tgt.x=max(self.radius_m*1.1,min(pot_tgt.x,CFG["ARENA_WIDTH_M"]-self.radius_m*1.1)) 
                pot_tgt.y=max(self.radius_m*1.1,min(pot_tgt.y,CFG["ARENA_HEIGHT_M"]-self.radius_m*1.1))
                all_others = opps + [t for t in game_state.get('teammates',[]) if t is not self]
                clr = self.radius_m+CFG["BALL_RADIUS_M"]+CFG.get("AI_DRIBBLE_PATH_EXTRA_CLEARANCE_M",0.03)
                if self._is_path_clear(self.pos_m, pot_tgt, all_others, clr):
                    min_dist_sq_to_any_player = float('inf')
                    for p_o in all_others: min_dist_sq_to_any_player=min(min_dist_sq_to_any_player, pot_tgt.distance_squared_to(p_o.pos_m))
                    open_score = math.sqrt(min_dist_sq_to_any_player) 
                    if dir_goal_base.length_squared()>0: open_score += safe_normalize(pot_tgt-self.pos_m).dot(dir_goal_base)*CFG.get("AI_DRIBBLE_GOAL_DIRECTION_BONUS",0.3) 
                    if open_score > max_open: max_open,best_tgt = open_score,pot_tgt
            if best_tgt is None: 
                default_dir = safe_normalize(opp_goal-self.pos_m) if (opp_goal-self.pos_m).length_squared()>1e-6 else pygame.math.Vector2(1 if self.team=='A' else -1,0) 
                self.dribble_target_m = self.pos_m + default_dir * (self.radius_m * 2.0)
                act_name+="_Default"
            else: self.dribble_target_m = best_tgt; act_name+="_NewTgt"
            self.dribble_target_persists_frames=CFG.get("AI_DRIBBLE_PERSIST_FRAMES",8)
            self._set_steering_target_for_action(self.dribble_target_m,"AI_Slowing_RADIUS_BALL",act_name)
        else: self._set_steering_target_for_action(self.dribble_target_m,"AI_Slowing_RADIUS_BALL",act_name+"_Persist")
        self.dribble_target_persists_frames-=1
        if self._is_at_target(target_threshold_factor=CFG["AI_ARRIVAL_THRESHOLD_FACTOR"]*0.8): 
            self.dribble_target_m=None;self.dribble_target_persists_frames=0;return "SUCCESS"
        return "RUNNING"
    def _move_to_position_action(self,target_pos,action_name,game_state,slowing_radius_key="AI_Slowing_RADIUS_POSITION"): 
        target_changed = True
        if self.primary_steering_target_m and self.current_action_name_for_debug==action_name:
             if self.primary_steering_target_m.distance_squared_to(target_pos)<=(self.radius_m*CFG.get("AI_POSITIONING_TARGET_UPDATE_THRESHOLD_FACTOR",0.5))**2:
                 target_changed = False
        if target_changed: self._set_steering_target_for_action(target_pos,slowing_radius_key,action_name)
        if self._is_at_target(): return "SUCCESS"
        return "RUNNING"
    def action_MoveToAttackingSupport(self, game_state): 
        ball,teammates,_,opp_goal,_,_ = self._get_game_context_objects(game_state)
        has_tm_ball = self.condition_TeammateHasBall(game_state) and teammates
        ref_pos = teammates[0].pos_m if has_tm_ball else ball.pos_m
        x_min = CFG["ARENA_WIDTH_M"]*CFG["AI_ATTACKER_OFFBALL_X_ZONE_MIN_FACTOR"]
        x_max = CFG["ARENA_WIDTH_M"]*CFG["AI_ATTACKER_OFFBALL_X_ZONE_MAX_FACTOR"]
        x_off = CFG["PLAYER_RADIUS_M"]*(7 if has_tm_ball else 4) 
        tgt_x = ref_pos.x - x_off if self.team=='B' else ref_pos.x + x_off 
        tgt_x = max(x_min, min(tgt_x, x_max)) 
        y_spread = CFG["ARENA_HEIGHT_M"]*CFG["AI_ATTACKER_OFFBALL_Y_SPREAD_FACTOR"]
        if ref_pos.y < CFG["ARENA_HEIGHT_M"]/3: tgt_y = CFG["ARENA_HEIGHT_M"]*0.66 
        elif ref_pos.y > CFG["ARENA_HEIGHT_M"]*2/3: tgt_y = CFG["ARENA_HEIGHT_M"]*0.33
        else: tgt_y = self.pos_m.y + (y_spread if self.pos_m.y < CFG["ARENA_HEIGHT_M"]/2 else -y_spread) 
        tgt_y = max(self.radius_m*2, min(tgt_y, CFG["ARENA_HEIGHT_M"]-self.radius_m*2)) 
        final_target = pygame.math.Vector2(tgt_x,tgt_y)
        return self._move_to_position_action(final_target,"MoveToAttackingSupport",game_state)
    def action_MoveToDefensiveCover(self, game_state): 
        ball,_,opps,my_goal,_,_ = self._get_game_context_objects(game_state)
        act_name_base = "MoveToDefensiveCover"
        recalc = self.defensive_cover_target_m is None or self.defensive_cover_persists_frames<=0 or \
                 (self.defensive_cover_target_m and ball.pos_m.distance_squared_to(self.defensive_cover_target_m)>(CFG["ARENA_WIDTH_M"]*0.3)**2)
        if recalc:
            dir_ball_mygoal = safe_normalize(my_goal-ball.pos_m);
            if dir_ball_mygoal.length_squared()<1e-6: dir_ball_mygoal=pygame.math.Vector2(-1 if self.team=='B' else 1,0)
            final_tgt_pos = ball.pos_m + dir_ball_mygoal*CFG["AI_DEFENDER_INTERCEPT_STANDOFF_M"]
            most_threat,min_threat_score=None,float('inf')
            for opp in opps:
                if safe_normalize(opp.pos_m-ball.pos_m).dot(dir_ball_mygoal) > CFG.get("AI_DEF_COVER_THREAT_CONE_DOT",0.1) and \
                   opp.pos_m.distance_squared_to(ball.pos_m) < (CFG["ARENA_WIDTH_M"]*0.35)**2:
                    score = opp.pos_m.distance_squared_to(my_goal)*0.7 + opp.pos_m.distance_squared_to(ball.pos_m)*0.3
                    if score < min_threat_score: min_threat_score,most_threat=score,opp
            if most_threat: final_tgt_pos = final_tgt_pos*0.4 + (most_threat.pos_m + safe_normalize(my_goal-most_threat.pos_m)*(self.radius_m*2.0))*0.6
            else: final_tgt_pos = ball.pos_m + dir_ball_mygoal*(CFG["AI_DEFENDER_INTERCEPT_STANDOFF_M"]+CFG["AI_DEFENDER_COVER_SPACE_OFFSET_M"])
            min_x_b = CFG["ARENA_WIDTH_M"]*CFG.get("AI_DEFENDER_MIN_X_B_FACTOR",0.45)
            max_x_a = CFG["ARENA_WIDTH_M"]*CFG.get("AI_DEFENDER_MAX_X_A_FACTOR",0.55)
            if self.team=='B': final_tgt_pos.x = max(min_x_b,min(final_tgt_pos.x,my_goal.x-self.radius_m*1.0))
            else: final_tgt_pos.x = min(max_x_a,max(final_tgt_pos.x,my_goal.x+self.radius_m*1.0))
            final_tgt_pos.y = max(self.radius_m*1.5,min(final_tgt_pos.y,CFG["ARENA_HEIGHT_M"]-self.radius_m*1.5))
            self.defensive_cover_target_m,self.defensive_cover_persists_frames = final_tgt_pos,CFG.get("AI_DEF_COVER_PERSIST_FRAMES",6)
            act_name_base+="_NewTgt"
        else: act_name_base+="_Persist"
        self.defensive_cover_persists_frames-=1
        return self._move_to_position_action(self.defensive_cover_target_m,act_name_base,game_state)
    def action_HoldMidfieldLine(self, game_state): 
        ball,_,_,_,_,_ = self._get_game_context_objects(game_state)
        tgt_x_factor = CFG["AI_DEFENDER_MIDFIELD_HOLD_X_FACTOR_B"] if self.team=='B' else CFG["AI_DEFENDER_MIDFIELD_HOLD_X_FACTOR_A"]
        tgt_x = CFG["ARENA_WIDTH_M"]*tgt_x_factor
        cur_tgt_y = self.primary_steering_target_m.y if self.primary_steering_target_m and self.current_action_name_for_debug=="HoldMidfieldLine" else CFG["ARENA_HEIGHT_M"]/2
        tgt_y_ball = (ball.pos_m.y*0.4) + (CFG["ARENA_HEIGHT_M"]/2*0.6)
        if abs(ball.pos_m.y-self.last_ball_y_for_midfield_hold) > CFG.get("AI_MIDFIELD_HOLD_BALL_Y_DEADZONE_M",0.25) or self.last_ball_y_for_midfield_hold<0:
            tgt_y,self.last_ball_y_for_midfield_hold = tgt_y_ball,ball.pos_m.y
        else: tgt_y = cur_tgt_y
        tgt_y = max(CFG["ARENA_HEIGHT_M"]*0.25,min(tgt_y,CFG["ARENA_HEIGHT_M"]*0.75))
        return self._move_to_position_action(pygame.math.Vector2(tgt_x,tgt_y),"HoldMidfieldLine",game_state,"AI_Slowing_RADIUS_POSITION")

    def _build_behavior_tree(self): 
        p = self 
        if self.player_num == 1: 
            self.behavior_tree = Selector(f"ROOT_Attacker_B{p.player_num}", player=p, children=[
                Sequence("OffensivePlayWithBall", player=p, children=[
                    ConditionNode("HasBall", player=p),
                    Selector("ChooseOffensiveActionWithBall", player=p, children=[
                        Sequence("EscapeOwnThird_Attacker", player=p, children=[ ConditionNode("BallInMyDefensiveThird", player=p), Selector("AttackerEscapeChoice", player=p, children=[ ActionNode("ClearBall", player=p), ActionNode("DribbleStrategically", player=p)])]),
                        Sequence("TryShoot", player=p, children=[ConditionNode("CanShoot", player=p), ActionNode("ShootAtGoal", player=p)]),
                        Sequence("TryPass", player=p, children=[ConditionNode("CanPassToTeammate", player=p), ActionNode("PassToTeammate", player=p)]),
                        ActionNode("DribbleStrategically", player=p)
                    ])]),
                Sequence("WinBall_Attacker", player=p, children=[ ConditionNode("IsDesignatedBallWinner", player=p), ConditionNode("HasBall", player=p, negate=True), ConditionNode("TeammateHasBall", player=p, negate=True), ActionNode("GoToBall", player=p)]),
                Sequence("SupportOffense_Attacker", player=p, children=[ ConditionNode("HasBall", player=p, negate=True), ConditionNode("IsDesignatedBallWinner", player=p, negate=True), ActionNode("MoveToAttackingSupport", player=p)]),
                ActionNode("GoToBall", player=p) 
            ])
        elif self.player_num == 2: 
            self.behavior_tree = Selector(f"ROOT_Defender_B{p.player_num}", player=p, children=[
                Sequence("DefensivePlayWithBall", player=p, children=[ConditionNode("HasBall", player=p),ActionNode("ClearBall", player=p)]),
                Sequence("InterceptHighPriorityThreat_Defender", player=p, children=[ ConditionNode("IsDesignatedBallWinner", player=p), ConditionNode("IsMyGoalThreatened", player=p), ConditionNode("HasBall", player=p, negate=True), ActionNode("GoToBall", player=p)]),
                Sequence("CoverSpace_Defender", player=p, children=[ ConditionNode("IsDesignatedBallWinner", player=p, negate=True), ConditionNode("HasBall", player=p, negate=True), Selector("WhenToCover", player=p, children=[ ConditionNode("IsMyGoalThreatened", player=p), ConditionNode("BallInMyDefensiveThird", player=p), ]), ActionNode("MoveToDefensiveCover", player=p)]),
                Sequence("HoldMidfieldLine_Defender", player=p, children=[ ConditionNode("BallInOpponentHalf", player=p), ConditionNode("IsMyGoalThreatened", player=p, negate=True), ActionNode("HoldMidfieldLine", player=p)]),
                ActionNode("MoveToDefensiveCover", player=p) 
            ])

    def update_ai(self, game_state_for_bt): 
        if not self.is_ai_driven_by_bt or not self.behavior_tree:

            super().update(game_state_for_bt['dt'])
            return

        dt = game_state_for_bt['dt']
        if self.kick_cooldown_timer > 0: self.kick_cooldown_timer -=1

        current_has_ball = self.condition_HasBall(game_state_for_bt)
        just_gained_ball = current_has_ball and not self._last_has_ball_status
        just_lost_ball = not current_has_ball and self._last_has_ball_status
        designation_changed = hasattr(self,'_last_designated_winner_status') and \
                              self._last_designated_winner_status != game_state_for_bt['is_designated_ball_winner']
        if just_gained_ball or just_lost_ball or designation_changed:
            self.dribble_target_m=None; self.dribble_target_persists_frames=0
            self.defensive_cover_target_m=None; self.defensive_cover_persists_frames=0
            log_ai(self,"EVENT",f"StateChg:GainB={just_gained_ball},LostB={just_lost_ball},DesigCh={designation_changed}->Reset BT.")
            self.behavior_tree.reset()
        self._last_has_ball_status = current_has_ball
        self._last_designated_winner_status = game_state_for_bt['is_designated_ball_winner']

        player_focus_check = (CFG.get("DEBUG_AI_PLAYER_FOCUS_NUM",0) in [0, self.player_num] and self.team == 'B')
        if player_focus_check: log_ai(self,"TICK_START",f"AI HasB:{current_has_ball},IsWin:{self._last_designated_winner_status},PrevAct:{self.current_action_name_for_debug},Cool:{self.kick_cooldown_timer}")

        bt_status = self.behavior_tree.tick(game_state_for_bt)

        force_objective = pygame.math.Vector2(0,0)
        if self.primary_steering_target_m:
            force_objective = self._arrive(self.primary_steering_target_m, self.slowing_radius_for_primary)
        force_separation = pygame.math.Vector2(0,0)
        if game_state_for_bt['teammates']: 
            force_separation = self._separation(game_state_for_bt['teammates'], CFG["AI_SEPARATION_DISTANCE"])

        if player_focus_check and CFG.get("DEBUG_LOG_AI_STEERING",False): log_ai(self,"STEERING_FORCES",f"ObjF:{force_objective.length():.2f},SepF:{force_separation.length():.2f}")

        current_steering_force = force_objective*CFG["AI_WEIGHT_PRIMARY_OBJECTIVE"] + force_separation*CFG["AI_WEIGHT_SEPARATION"]
        self._apply_steering_force(current_steering_force) 

        if player_focus_check: log_ai(self,"TICK_END",f"NewAct:{self.current_action_name_for_debug},BT:{bt_status},Tgt:{self.primary_steering_target_m if self.primary_steering_target_m else 'None'},Vel:{self.vel_m_s.length():.2f}")

        super().update(dt)

class Ball(Entity): 
    def __init__(self, pos_m):
        super().__init__(pos_m, CFG["BALL_RADIUS_M"], CFG["BALL_MASS"], CFG["COLOR_BALL"])
        self.damping = CFG["BALL_DAMPING"]
    def update(self, dt):

        super().update(dt)

        if self.vel_m_s.length_squared() < (0.001*0.001): 
            self.vel_m_s.update(0,0)

def apriltag_processing_loop():
    global camera_matrix_at, dist_coeffs_at
    global perspective_matrix_metric_to_sim_at, perspective_matrix_pixel_to_sim_at
    global apriltag_detector_at, camera_capture_at
    global TAG_SIZE_METERS, ALL_ROBOT_TAG_IDS, LOWER_YELLOW_HSV, UPPER_YELLOW_HSV
    global BALL_MORPH_KERNEL_SIZE, BALL_ERODE_ITERATIONS, BALL_DILATE_ITERATIONS
    global MIN_BALL_CONTOUR_AREA_PX, MIN_BALL_CIRCULARITY

    if apriltag_detector_at is None or \
       camera_capture_at is None or \
       camera_matrix_at is None or \
       dist_coeffs_at is None or \
       perspective_matrix_metric_to_sim_at is None or \
       perspective_matrix_pixel_to_sim_at is None:
        print_once("at_thread_init_fail_critical", "AT_THREAD (UltimateSim): Critical components (incl. M_pixel_to_sim) not initialized. Exiting.")
        return

    fx, fy = camera_matrix_at[0,0], camera_matrix_at[1,1]
    cx, cy = camera_matrix_at[0,2], camera_matrix_at[1,2]
    cam_params_for_pose_estimation = (fx,fy,cx,cy)

    actual_w = int(camera_capture_at.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(camera_capture_at.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"AT_THREAD (UltimateSim): Starting detection. Cam res: {actual_w}x{actual_h}")

    ball_morph_kernel = np.ones((BALL_MORPH_KERNEL_SIZE, BALL_MORPH_KERNEL_SIZE), np.uint8)

    SIM_ORIENT_FRONT_INDICES = (0, 1)
    SIM_ORIENT_BACK_INDICES = (3, 2)

    while not stop_apriltag_thread.is_set():
        ret, frame = camera_capture_at.read()
        if not ret:
            print_once("at_thread_cam_fail","AT_THREAD (UltimateSim): Frame grab fail."); time.sleep(0.1); continue
        if frame.shape[1]!=actual_w or frame.shape[0]!=actual_h: 
            print_once("at_thread_res_chg",f"AT_THREAD (UltimateSim): Frame res {frame.shape[1]}x{frame.shape[0]} != expected {actual_w}x{actual_h}. Skipping."); time.sleep(0.1); continue

        try:
            undistorted_frame = cv2.undistort(frame, camera_matrix_at, dist_coeffs_at, None, camera_matrix_at)

            ball_sim_pos_update = None
            hsv_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv_frame, LOWER_YELLOW_HSV, UPPER_YELLOW_HSV)

            processed_mask = yellow_mask
            if BALL_ERODE_ITERATIONS > 0:
                processed_mask = cv2.erode(processed_mask, ball_morph_kernel, iterations=BALL_ERODE_ITERATIONS)
            if BALL_DILATE_ITERATIONS > 0:
                processed_mask = cv2.dilate(processed_mask, ball_morph_kernel, iterations=BALL_DILATE_ITERATIONS)

            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_ball_contour = None
            if contours:
                valid_ball_candidates = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > MIN_BALL_CONTOUR_AREA_PX:
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 1e-3:
                            circularity = 4 * np.pi * (area / (perimeter * perimeter))
                            if circularity > MIN_BALL_CIRCULARITY:
                                valid_ball_candidates.append(contour)
                if valid_ball_candidates:
                    best_ball_contour = max(valid_ball_candidates, key=cv2.contourArea)

            if best_ball_contour is not None:
                ((cX_px, cY_px), radius_px) = cv2.minEnclosingCircle(best_ball_contour)
                pt_pixel_ball = np.array([[[cX_px, cY_px]]], dtype=np.float32)
                pt_sim_ball = cv2.perspectiveTransform(pt_pixel_ball, perspective_matrix_pixel_to_sim_at)
                if pt_sim_ball is not None:
                    sim_x, sim_y = pt_sim_ball[0][0][0], pt_sim_ball[0][0][1]
                    ball_sim_pos_update = (sim_x, sim_y)

            gray_frame_for_tags = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
            detections = apriltag_detector_at.detect(
                gray_frame_for_tags, 
                estimate_tag_pose=True,
                camera_params=cam_params_for_pose_estimation,
                tag_size=TAG_SIZE_METERS
            )

            robot_positions_update = {}
            robot_orientations_update = {}

            for tag in detections:
                if tag.tag_id in ALL_ROBOT_TAG_IDS:
                    if tag.pose_t is not None:
                        cam_x,cam_y = tag.pose_t.flatten()[0],tag.pose_t.flatten()[1]
                        pt_cam_robot = np.array([[[cam_x,cam_y]]],dtype=np.float32)
                        pt_sim_robot = cv2.perspectiveTransform(pt_cam_robot, perspective_matrix_metric_to_sim_at)
                        if pt_sim_robot is not None:
                            robot_positions_update[tag.tag_id] = (pt_sim_robot[0][0][0],pt_sim_robot[0][0][1])

                    if tag.corners is not None:
                        pixel_corners = tag.corners.reshape(-1, 1, 2).astype(np.float32)
                        sim_corners_m = cv2.perspectiveTransform(pixel_corners, perspective_matrix_pixel_to_sim_at)
                        if sim_corners_m is not None:
                            orientation_deg = sandbox_utils.calculate_orientation_from_sim_corners(
                                sim_corners_m,
                                front_indices=SIM_ORIENT_FRONT_INDICES,
                                back_indices=SIM_ORIENT_BACK_INDICES
                            )
                            if orientation_deg is not None:
                                robot_orientations_update[tag.tag_id] = orientation_deg

            update_payload = {
                'ball_sim_pos': ball_sim_pos_update,
                'robot_positions': robot_positions_update,
                'robot_orientations': robot_orientations_update
            }

            if apriltag_queue.full():
                try: apriltag_queue.get_nowait() 
                except queue.Empty: pass
            apriltag_queue.put_nowait(update_payload)

            time.sleep(0.001)
        except Exception as e:
            print_once(f"at_err_loop_{type(e).__name__}", f"AT_THREAD (UltimateSim) ERR in loop: {e}")

            time.sleep(0.5)

    print("AT_THREAD (UltimateSim): Loop stopped.")
    if camera_capture_at:
        camera_capture_at.release()
        print("AT_THREAD (UltimateSim): Camera released.")

class Game:
    def __init__(self):
        global camera_matrix_at, dist_coeffs_at, perspective_matrix_metric_to_sim_at, perspective_matrix_pixel_to_sim_at
        global apriltag_detector_at, camera_capture_at, apriltag_thread, _ai_log_file_handle
        global DETECTOR_KWARGS, APRILTAG_FAMILY, CAMERA_INDEX, APRILTAG_CAMERA_RESOLUTION_W, APRILTAG_CAMERA_RESOLUTION_H
        global CALIBRATION_DATA_FOLDER, CAMERA_CALIBRATION_FILE_TEMPLATE, PERSPECTIVE_MATRICES_NPZ_FILE

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((CFG["SCREEN_WIDTH"],CFG["SCREEN_HEIGHT"]))
        pygame.display.set_caption("Main 4-Robot + Ball AprilTag Simulation")
        self.clock = pygame.time.Clock()
        self.font_score = pygame.font.SysFont("Arial",36,bold=True)
        self.font_message = pygame.font.SysFont("Arial",60,bold=True)

        if CFG["DEBUG_LOG_AI_GENERAL"]:
            try:
                _ai_log_file_handle = open(_AI_LOG_FILE_PATH,"w",encoding="utf-8")
                _ai_log_file_handle.write(f"--- AI LOG SESSION START @ {time.asctime()} ---\n")
                _ai_log_file_handle.flush()
                print(f"AI Logging to: {_AI_LOG_FILE_PATH}")
            except Exception as e:
                print(f"ERROR: Could not open AI log file '{_AI_LOG_FILE_PATH}': {e}")
                _ai_log_file_handle = None

        print("GAME INIT: Loading AprilTag & Calibration data...")
        calib_file_name = CAMERA_CALIBRATION_FILE_TEMPLATE.format(width=APRILTAG_CAMERA_RESOLUTION_W,height=APRILTAG_CAMERA_RESOLUTION_H)
        calib_file_path = os.path.join(CALIBRATION_DATA_FOLDER, calib_file_name)
        try:
            calib_data = np.load(calib_file_path)
            camera_matrix_at = calib_data['camera_matrix']
            dist_coeffs_at = calib_data['dist_coeffs']
            loaded_w, loaded_h = int(calib_data.get('image_width',0)), int(calib_data.get('image_height',0))
            if loaded_w != APRILTAG_CAMERA_RESOLUTION_W or loaded_h != APRILTAG_CAMERA_RESOLUTION_H:
                 print_once("calib_res_warn_load", f"WARNING: Loaded calib file '{calib_file_path}' is for {loaded_w}x{loaded_h}, but configured for {APRILTAG_CAMERA_RESOLUTION_W}x{APRILTAG_CAMERA_RESOLUTION_H}.")
            print(f"  Loaded camera calibration from: {calib_file_path}")
        except Exception as e:
            raise RuntimeError(f"FATAL ERROR: Could not load camera calibration file '{calib_file_path}': {e}. Please run setup.py.")

        try:
            pt_data = np.load(PERSPECTIVE_MATRICES_NPZ_FILE)
            perspective_matrix_metric_to_sim_at = pt_data['M_metric_to_sim']
            perspective_matrix_pixel_to_sim_at = pt_data['M_pixel_to_sim'] 

            stored_res_pt = pt_data.get('camera_resolution_when_captured')
            if stored_res_pt is not None and (stored_res_pt[0]!=APRILTAG_CAMERA_RESOLUTION_W or stored_res_pt[1]!=APRILTAG_CAMERA_RESOLUTION_H) :
                print_once("pt_res_warn_load", f"WARNING: Perspective matrix file was for res {stored_res_pt}, but configured for {APRILTAG_CAMERA_RESOLUTION_W}x{APRILTAG_CAMERA_RESOLUTION_H}.")
            print(f"  Loaded perspective transform matrices (M_metric_to_sim and M_pixel_to_sim) from: {PERSPECTIVE_MATRICES_NPZ_FILE}")
        except FileNotFoundError:
             raise RuntimeError(f"FATAL ERROR: Perspective transform file '{PERSPECTIVE_MATRICES_NPZ_FILE}' not found. Please run setup.py.")
        except KeyError as e:
            raise RuntimeError(f"FATAL ERROR: Key '{e}' not found in perspective transform file '{PERSPECTIVE_MATRICES_NPZ_FILE}'. Ensure it contains 'M_metric_to_sim' and 'M_pixel_to_sim'. Please run setup.py.")
        except Exception as e:
            raise RuntimeError(f"FATAL ERROR: Could not load perspective transform file '{PERSPECTIVE_MATRICES_NPZ_FILE}': {e}. Please run setup.py.")

        try:
            apriltag_detector_at = pupil_apriltags.Detector(**DETECTOR_KWARGS)
            print(f"  AprilTag detector initialized for family: {APRILTAG_FAMILY}")
        except Exception as e:
            raise RuntimeError(f"FATAL ERROR: Could not initialize AprilTag Detector: {e}")

        camera_capture_at = cv2.VideoCapture(CAMERA_INDEX)
        if not camera_capture_at.isOpened():
            raise RuntimeError(f"FATAL ERROR: Cannot open camera {CAMERA_INDEX}")
        camera_capture_at.set(cv2.CAP_PROP_FRAME_WIDTH,APRILTAG_CAMERA_RESOLUTION_W)
        camera_capture_at.set(cv2.CAP_PROP_FRAME_HEIGHT,APRILTAG_CAMERA_RESOLUTION_H)
        time.sleep(0.5) 
        actual_w_at, actual_h_at = int(camera_capture_at.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera_capture_at.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w_at != APRILTAG_CAMERA_RESOLUTION_W or actual_h_at != APRILTAG_CAMERA_RESOLUTION_H:
            print_once("cam_res_final_warning",f"CRITICAL WARNING: Camera for AT thread started at {actual_w_at}x{actual_h_at}, but system is configured for {APRILTAG_CAMERA_RESOLUTION_W}x{APRILTAG_CAMERA_RESOLUTION_H}! This will cause issues.")
        print(f"  Camera for AprilTags opened at {actual_w_at}x{actual_h_at}")

        apriltag_thread = threading.Thread(target=apriltag_processing_loop, daemon=True)
        apriltag_thread.start()
        print("  AprilTag processing thread started.")

        self.sandbox_ws_initialized = False
        try:
            print("ULTIMATE_SIM: Initializing Sandbox WebSocket Client...")
            if sandbox_utils.initialize_sandbox_websocket_client():
                self.sandbox_ws_initialized = True
                print("ULTIMATE_SIM: Sandbox WebSocket Client initialized successfully.")
            else:
                print("ULTIMATE_SIM: Sandbox WebSocket Client initialization reported failure or not connected yet.")
        except Exception as e_ws_init:
            print(f"ULTIMATE_SIM: EXCEPTION during Sandbox WebSocket Client initialization: {e_ws_init}")
            traceback.print_exc()

        self.calculate_scales_and_pitch_geometry()
        self.players = []
        self.ball = None 
        self.score = {'A': 0, 'B': 0}
        self.tag_to_player_map = {} 
        self.game_active = True
        self.last_goal_time_ms = 0

        self.robot_orientations_deg = {} 

        self.setup_entities_for_kickoff()
        print("GAME INIT: Complete.")

    def calculate_scales_and_pitch_geometry(self):
        global METERS_TO_PIXELS, PITCH_RECT_PX
        drawable_width_px = CFG["SCREEN_WIDTH"] - 2 * CFG["MARGIN"]
        drawable_height_px = CFG["SCREEN_HEIGHT"] - 2 * CFG["MARGIN"]
        scale_from_width = drawable_width_px / CFG["ARENA_WIDTH_M"]
        scale_from_height = drawable_height_px / CFG["ARENA_HEIGHT_M"]
        METERS_TO_PIXELS = min(scale_from_width, scale_from_height)
        scaled_pitch_width_px = CFG["ARENA_WIDTH_M"] * METERS_TO_PIXELS
        scaled_pitch_height_px = CFG["ARENA_HEIGHT_M"] * METERS_TO_PIXELS
        pitch_x_px = (CFG["SCREEN_WIDTH"] - scaled_pitch_width_px) / 2
        pitch_y_px = (CFG["SCREEN_HEIGHT"] - scaled_pitch_height_px) / 2
        PITCH_RECT_PX = pygame.Rect(pitch_x_px, pitch_y_px, scaled_pitch_width_px, scaled_pitch_height_px)
        self.goal_y_min_m = (CFG["ARENA_HEIGHT_M"] - CFG["GOAL_WIDTH_M"]) / 2
        self.goal_y_max_m = (CFG["ARENA_HEIGHT_M"] + CFG["GOAL_WIDTH_M"]) / 2

    def get_initial_player_positions(self): 

        q_w = CFG["ARENA_WIDTH_M"] / 4.0  
        t_h1 = CFG["ARENA_HEIGHT_M"] / 3.0 
        t_h2 = CFG["ARENA_HEIGHT_M"] * 2.0 / 3.0 

        return [
            (ROBOT_A1_TAG_ID, (q_w, t_h1), 'A', 1, False), 
            (ROBOT_A2_TAG_ID, (q_w, t_h2), 'A', 2, False), 
            (ROBOT_B1_TAG_ID, (CFG["ARENA_WIDTH_M"] - q_w, t_h1), 'B', 1, True), 
            (ROBOT_B2_TAG_ID, (CFG["ARENA_WIDTH_M"] - q_w, t_h2), 'B', 2, True), 
        ]

    def setup_entities_for_kickoff(self):
        self.players.clear()
        self.tag_to_player_map.clear()
        initial_player_data = self.get_initial_player_positions()
        for tag_id,pos_tuple,team_char,num_role,is_ai in initial_player_data:
            player = Player(pygame.math.Vector2(pos_tuple),team_char,num_role,is_ai_driven_by_bt=is_ai)
            player.tag_id_link = tag_id
            self.players.append(player)
            self.tag_to_player_map[tag_id] = player

        self.ball = Ball(pygame.math.Vector2(CFG["ARENA_WIDTH_M"]/2,CFG["ARENA_HEIGHT_M"]/2))

        for p in self.players: 
            p.vel_m_s.update(0,0)
            p.acc_m_s2.update(0,0)
            if p.is_ai_driven_by_bt:
                if hasattr(p, 'behavior_tree') and p.behavior_tree: p.behavior_tree.reset()
                p.primary_steering_target_m = None
                p._last_has_ball_status = False
                p._last_designated_winner_status = False
                p.current_action_name_for_debug = "Idle_Kickoff_AI"
                p.kick_cooldown_timer = 0
                p.dribble_target_m = None; p.dribble_target_persists_frames = 0
                p.last_ball_y_for_midfield_hold = -1.0
                p.defensive_cover_target_m = None; p.defensive_cover_persists_frames = 0
            else: 
                p.current_action_name_for_debug = f"Idle_Kickoff_Human_A{p.player_num}"

        self.ball.vel_m_s.update(0,0)
        self.game_active=True
        log_ai(None,"GAME_EVENT","Kickoff setup complete for 4 players.")

    def handle_human_input(self):
        pass 

    def resolve_entity_collision(self,e1,e2,restitution):
        diff_vec = e1.pos_m - e2.pos_m; dist_m_sq = diff_vec.length_squared()
        min_dist_m = e1.radius_m + e2.radius_m
        if dist_m_sq < min_dist_m**2 and dist_m_sq > 1e-9:
            dist_m = math.sqrt(dist_m_sq); overlap = min_dist_m - dist_m
            separation_normal = diff_vec / dist_m
            total_inv_mass = (1/e1.mass if e1.mass > 1e-6 else 0) + (1/e2.mass if e2.mass > 1e-6 else 0)
            e1_move_factor, e2_move_factor = 0.5, 0.5
            if total_inv_mass > 1e-9:
                e1_move_factor = (1/e1.mass if e1.mass > 1e-6 else 0) / total_inv_mass
                e2_move_factor = (1/e2.mass if e2.mass > 1e-6 else 0) / total_inv_mass
            e1.pos_m += separation_normal * overlap * e1_move_factor
            e2.pos_m -= separation_normal * overlap * e2_move_factor
            relative_vel = e1.vel_m_s - e2.vel_m_s
            vel_along_normal = relative_vel.dot(separation_normal)
            if vel_along_normal < 0:
                impulse_scalar = -(1 + restitution) * vel_along_normal
                if total_inv_mass > 1e-9: impulse_scalar /= total_inv_mass
                else: impulse_scalar = 0
                impulse_vec = impulse_scalar * separation_normal
                if e1.mass > 1e-6 : e1.vel_m_s += impulse_vec / e1.mass
                if e2.mass > 1e-6 : e2.vel_m_s -= impulse_vec / e2.mass

    def update_game_state(self, dt):
        if not self.game_active:
            current_time_ms = pygame.time.get_ticks()
            if current_time_ms - self.last_goal_time_ms > CFG["RESET_DELAY_MS"]:
                if self.score['A'] >= CFG["MAX_SCORE"] or self.score['B'] >= CFG["MAX_SCORE"]:
                    self.score = {'A': 0, 'B': 0}
                    log_ai(None, "GAME_EVENT", "Max score reached, game fully reset.")
                self.setup_entities_for_kickoff()
            return

        latest_payload_from_thread = None
        try:

            while not apriltag_queue.empty():
                latest_payload_from_thread = apriltag_queue.get_nowait()
        except queue.Empty:
            pass 

        if latest_payload_from_thread:
            if self.ball and latest_payload_from_thread.get('ball_sim_pos') is not None:
                sim_x_ball, sim_y_ball = latest_payload_from_thread['ball_sim_pos']
                clamped_x_ball = max(self.ball.radius_m, min(sim_x_ball, CFG["ARENA_WIDTH_M"] - self.ball.radius_m))
                clamped_y_ball = max(self.ball.radius_m, min(sim_y_ball, CFG["ARENA_HEIGHT_M"] - self.ball.radius_m))
                self.ball.pos_m.update(clamped_x_ball, clamped_y_ball)
                self.ball.vel_m_s.update(0,0) 

            robot_positions = latest_payload_from_thread.get('robot_positions', {})
            for tag_id, (sim_x_robot, sim_y_robot) in robot_positions.items():
                if tag_id in self.tag_to_player_map:
                    player = self.tag_to_player_map[tag_id]
                    clamped_x_robot = max(player.radius_m, min(sim_x_robot, CFG["ARENA_WIDTH_M"] - player.radius_m))
                    clamped_y_robot = max(player.radius_m, min(sim_y_robot, CFG["ARENA_HEIGHT_M"] - player.radius_m))
                    player.pos_m.update(clamped_x_robot, clamped_y_robot)

                    if player.is_ai_driven_by_bt or player.tag_id_link is not None: 
                        player.vel_m_s.update(0,0)

            robot_orientations_from_at = latest_payload_from_thread.get('robot_orientations', {})
            for tag_id, orient_deg in robot_orientations_from_at.items():
                self.robot_orientations_deg[tag_id] = orient_deg

        ai_team_b_players = [p for p in self.players if p.team == 'B' and p.is_ai_driven_by_bt]
        designated_winner_b = None 
        if self.ball: 
            if len(ai_team_b_players) == 1: 
                designated_winner_b = ai_team_b_players[0]
            elif len(ai_team_b_players) > 1:

                p_att=next((p for p in ai_team_b_players if p.player_num==1),None)
                p_def=next((p for p in ai_team_b_players if p.player_num==2),None)
                if p_att and p_def:
                    d_att_sq = p_att.pos_m.distance_squared_to(self.ball.pos_m)
                    d_def_sq = p_def.pos_m.distance_squared_to(self.ball.pos_m)
                    adv_sq = CFG["AI_BALL_WINNER_PROXIMITY_ADVANTAGE_M"]**2
                    if d_att_sq < d_def_sq - adv_sq: designated_winner_b = p_att
                    elif d_def_sq < d_att_sq - adv_sq: designated_winner_b = p_def
                    else: designated_winner_b = p_att 
                elif p_att: designated_winner_b = p_att
                elif p_def: designated_winner_b = p_def

        if CFG.get("DEBUG_LOG_GAME_EVENTS"): 
            current_winner_str = str(designated_winner_b) if designated_winner_b else "None"
            last_winner_str = str(getattr(self, '_last_designated_b_winner', None)) if hasattr(self, '_last_designated_b_winner') else "None"
            if current_winner_str != last_winner_str :
                log_ai(None, "GAME_LOG", f"Designated B winner changed to: {current_winner_str}")
                self._last_designated_b_winner = designated_winner_b

        for p in self.players:
            if p.is_ai_driven_by_bt:
                teammates=[m for m in self.players if m.team==p.team and m is not p]
                opps=[o for o in self.players if o.team!=p.team]
                gs_bt={'ball':self.ball,'teammates':teammates,'opponents':opps,'dt':dt,
                       'is_designated_ball_winner':(p is designated_winner_b)}
                p.update_ai(gs_bt) 

                if p.team == 'B':
                    target_pos_m_vec = p.primary_steering_target_m
                    current_pos_m_vec = p.pos_m

                    if target_pos_m_vec and p.tag_id_link is not None:
                        current_orientation_deg = self.robot_orientations_deg.get(p.tag_id_link)

                        if current_orientation_deg is not None:
                            joy_x, joy_y = sandbox_utils.calculate_joystick_from_world_target(
                                current_pos_m=(current_pos_m_vec.x, current_pos_m_vec.y),
                                ai_target_pos_m=(target_pos_m_vec.x, target_pos_m_vec.y),
                                current_orientation_deg=current_orientation_deg
                            )
                            websocket_user_id = -1
                            if p.player_num == 1: websocket_user_id = 3
                            elif p.player_num == 2: websocket_user_id = 4

                            if self.sandbox_ws_initialized and websocket_user_id != -1:
                                sandbox_utils.send_transformed_joystick_command_ws(
                                    websocket_user_id=websocket_user_id,
                                    joy_x=joy_x,
                                    joy_y=joy_y
                                )
                        else:
                            error_key = f"no_orient_B{p.player_num}_tag{p.tag_id_link}"
                            print_once(error_key, f"ULTIMATE_SIM: No orientation for B{p.player_num} (Tag {p.tag_id_link}) to send to WS.")
            else: 

                p.update(dt) 

        if self.ball:
            if not (latest_payload_from_thread and latest_payload_from_thread.get('ball_sim_pos') is not None):
                self.ball.update(dt) 

        for p_entity in self.players + ([self.ball] if self.ball else []):

            is_player_entity = isinstance(p_entity, Player)

            if p_entity.pos_m.x - p_entity.radius_m < 0:
                p_entity.pos_m.x = p_entity.radius_m
                if is_player_entity and p_entity.vel_m_s.x < 0: p_entity.vel_m_s.x *= -0.5 
                elif not is_player_entity and p_entity.vel_m_s.x < 0: 

                    if not (self.goal_y_min_m < p_entity.pos_m.y < self.goal_y_max_m):
                        p_entity.vel_m_s.x *= -CFG["RESTITUTION_BALL_WALL"]

            elif p_entity.pos_m.x + p_entity.radius_m > CFG["ARENA_WIDTH_M"]:
                p_entity.pos_m.x = CFG["ARENA_WIDTH_M"] - p_entity.radius_m
                if is_player_entity and p_entity.vel_m_s.x > 0: p_entity.vel_m_s.x *= -0.5 
                elif not is_player_entity and p_entity.vel_m_s.x > 0: 
                    if not (self.goal_y_min_m < p_entity.pos_m.y < self.goal_y_max_m):
                         p_entity.vel_m_s.x *= -CFG["RESTITUTION_BALL_WALL"]

            if p_entity.pos_m.y - p_entity.radius_m < 0:
                p_entity.pos_m.y = p_entity.radius_m
                if p_entity.vel_m_s.y < 0: 
                    p_entity.vel_m_s.y *= (-0.5 if is_player_entity else -CFG["RESTITUTION_BALL_WALL"])
            elif p_entity.pos_m.y + p_entity.radius_m > CFG["ARENA_HEIGHT_M"]:
                p_entity.pos_m.y = CFG["ARENA_HEIGHT_M"] - p_entity.radius_m
                if p_entity.vel_m_s.y > 0:
                    p_entity.vel_m_s.y *= (-0.5 if is_player_entity else -CFG["RESTITUTION_BALL_WALL"])

        scored_this_frame = False; goal_scorer_team_char = None
        if self.ball:

            if self.ball.pos_m.x - self.ball.radius_m <= 0 and \
               self.goal_y_min_m < self.ball.pos_m.y < self.goal_y_max_m:
                self.score['B'] += 1; scored_this_frame = True; goal_scorer_team_char = 'B'

            elif self.ball.pos_m.x + self.ball.radius_m >= CFG["ARENA_WIDTH_M"] and \
                 self.goal_y_min_m < self.ball.pos_m.y < self.goal_y_max_m:
                self.score['A'] += 1; scored_this_frame = True; goal_scorer_team_char = 'A'

            self.ball.pos_m.x = max(0 - self.ball.radius_m - CFG["GOAL_DEPTH_M"], min(self.ball.pos_m.x, CFG["ARENA_WIDTH_M"] + self.ball.radius_m + CFG["GOAL_DEPTH_M"]))
            self.ball.pos_m.y = max(self.ball.radius_m, min(self.ball.pos_m.y, CFG["ARENA_HEIGHT_M"] - self.ball.radius_m))

        if scored_this_frame: 
            self.game_active=False; self.last_goal_time_ms=pygame.time.get_ticks()
            goal_message_t = f"GOAL! Team {goal_scorer_team_char} scored. Score A:{self.score['A']} B:{self.score['B']}"
            print(goal_message_t)
            log_ai(None,"GAME_EVENT",goal_message_t)
            for p_reset in self.players:
                if p_reset.is_ai_driven_by_bt and hasattr(p_reset, 'behavior_tree') and p_reset.behavior_tree:
                    p_reset.behavior_tree.reset(); p_reset._last_has_ball_status = False; p_reset._last_designated_winner_status = False
                    p_reset.current_action_name_for_debug = "Idle_GoalReset_AI"; p_reset.kick_cooldown_timer = 0
                    p_reset.dribble_target_m = None; p_reset.dribble_target_persists_frames = 0
                    p_reset.last_ball_y_for_midfield_hold = -1.0; p_reset.defensive_cover_target_m = None; p_reset.defensive_cover_persists_frames = 0

        for i in range(len(self.players)):
            for j in range(i+1, len(self.players)):
                self.resolve_entity_collision(self.players[i], self.players[j], CFG["RESTITUTION_PLAYER_PLAYER"])
            if self.ball:
                self.resolve_entity_collision(self.players[i], self.ball, CFG["RESTITUTION_PLAYER_BALL"])

    def draw_pitch_and_field(self):
        self.screen.fill(CFG["COLOR_BACKGROUND"])
        pygame.draw.rect(self.screen, CFG["COLOR_PITCH"], PITCH_RECT_PX)
        pygame.draw.rect(self.screen, CFG["COLOR_LINES"], PITCH_RECT_PX, LINE_THICKNESS_PX)
        pygame.draw.line(self.screen, CFG["COLOR_LINES"], (PITCH_RECT_PX.centerx,PITCH_RECT_PX.top), (PITCH_RECT_PX.centerx,PITCH_RECT_PX.bottom), LINE_THICKNESS_PX)
        pygame.draw.circle(self.screen, CFG["COLOR_LINES"], PITCH_RECT_PX.center, int(CFG["ARENA_HEIGHT_M"]*0.15*METERS_TO_PIXELS), LINE_THICKNESS_PX)
        gw_px = CFG["GOAL_WIDTH_M"]*METERS_TO_PIXELS
        gty_px = PITCH_RECT_PX.centery - gw_px/2
        gby_px = PITCH_RECT_PX.centery + gw_px/2
        pygame.draw.line(self.screen,CFG["COLOR_GOAL"],(PITCH_RECT_PX.left,gty_px),(PITCH_RECT_PX.left,gby_px),GOAL_LINE_THICKNESS_PX)
        pygame.draw.line(self.screen,CFG["COLOR_GOAL"],(PITCH_RECT_PX.right,gty_px),(PITCH_RECT_PX.right,gby_px),GOAL_LINE_THICKNESS_PX)
        gd_px = CFG["GOAL_DEPTH_M"]*METERS_TO_PIXELS
        pygame.draw.rect(self.screen,CFG["COLOR_LINES"],(PITCH_RECT_PX.left-gd_px,gty_px,gd_px,gw_px),LINE_THICKNESS_PX//2)
        pygame.draw.rect(self.screen,CFG["COLOR_LINES"],(PITCH_RECT_PX.right,gty_px,gd_px,gw_px),LINE_THICKNESS_PX//2)

    def draw_scores_and_messages(self):
        score_text = f"Team A: {self.score['A']}  -  Team B: {self.score['B']}"
        text_surface = self.font_score.render(score_text, True, CFG["COLOR_LINES"])
        text_rect = text_surface.get_rect(center=(CFG["SCREEN_WIDTH"]/2, CFG["MARGIN"]/2))
        self.screen.blit(text_surface, text_rect)
        if not self.game_active:
            message = "GOAL!"
            small_message = "" 
            if self.score['A']>=CFG["MAX_SCORE"] or self.score['B']>=CFG["MAX_SCORE"]:
                winner = "Team A" if self.score['A']>=CFG["MAX_SCORE"] else "Team B"
                message = f"{winner} WINS!"
            msg_surface = self.font_message.render(message, True, (255,215,0))
            msg_rect = msg_surface.get_rect(center=(CFG["SCREEN_WIDTH"]/2, CFG["SCREEN_HEIGHT"]/2 - 20))
            self.screen.blit(msg_surface, msg_rect)

    def render_all(self):
        self.draw_pitch_and_field()
        for p in self.players: p.draw(self.screen)
        if self.ball: self.ball.draw(self.screen)
        self.draw_scores_and_messages()
        pygame.display.flip()

    def game_loop(self):
        global stop_apriltag_thread, apriltag_thread
        is_running = True
        print("GAME LOOP: Started.")
        try:
            while is_running:
                dt = min(self.clock.tick(CFG["FPS"])/1000.0, 0.1) 
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                       (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        is_running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("Manual reset (R pressed).")
                        log_ai(None, "GAME_EVENT", "Manual reset (R pressed).")
                        self.score = {'A':0,'B':0} 
                        self.last_goal_time_ms = 0 
                        self.game_active = False 

                self.handle_human_input() 
                self.update_game_state(dt)
                self.render_all()
        finally:
            print("GAME LOOP: Exiting. Signaling AprilTag thread to stop...")
            stop_apriltag_thread.set()
            if apriltag_thread and apriltag_thread.is_alive():
                print("GAME LOOP: Waiting for AprilTag thread to join...")
                apriltag_thread.join(timeout=3.0)
                if apriltag_thread.is_alive():
                    print("GAME LOOP: WARNING - AprilTag thread did not stop in time. Forcing exit.")
                else:
                    print("GAME LOOP: AprilTag thread joined successfully.")

            if hasattr(self, 'sandbox_ws_initialized') and self.sandbox_ws_initialized:
                try:
                    print("ULTIMATE_SIM: Shutting down Sandbox WebSocket Client...")
                    sandbox_utils.shutdown_sandbox_websocket_client()
                    print("ULTIMATE_SIM: Sandbox WebSocket Client shutdown command issued.")
                except Exception as e_ws_shutdown:
                    print(f"ULTIMATE_SIM: EXCEPTION during Sandbox WebSocket Client shutdown: {e_ws_shutdown}")
                    traceback.print_exc()

            if pygame.get_init(): 
                pygame.quit()
                print("GAME LOOP: Pygame quit successfully.")

            global _ai_log_file_handle 
            if _ai_log_file_handle and not _ai_log_file_handle.closed:
                _ai_log_file_handle.write(f"--- AI LOG SESSION END @ {time.asctime()} (Game Loop Exit) ---\n")
                _ai_log_file_handle.close()
                if os.path.exists(_AI_LOG_FILE_PATH): print(f"AI Log closed. See: {_AI_LOG_FILE_PATH}")
                _ai_log_file_handle = None

if __name__ == '__main__':

    try: 
        del _PLAYER_RADIUS_M_VAL,_BALL_RADIUS_M_VAL,_ARENA_WIDTH_M_VAL,_ARENA_HEIGHT_M_VAL,_GOAL_WIDTH_M_VAL
    except NameError: 
        pass

    game_instance = None
    try:
        print("Main: Initializing 4-Robot + Ball AprilTag Simulation...")
        game_instance = Game() 
        print("Main: Starting game loop...")
        game_instance.game_loop() 
    except RuntimeError as e: 
        print(f"INITIALIZATION ERROR: {e}")
        print("Please ensure setup.py has been run successfully and all .npz files are present and correct for the configured resolution.")

    except Exception as e: 
        print(f"UNEXPECTED CRITICAL ERROR IN MAIN: {e}")
        traceback.print_exc()
    finally:

        if not (game_instance and hasattr(game_instance, 'clock') and game_instance.clock is not None):
            print_once("main_final_cleanup_check_at", "Main(finally): Game instance/loop might not have run. Ensuring AT thread stop signal.")
            stop_apriltag_thread.set() 
            if apriltag_thread and apriltag_thread.is_alive():
                print("Main(finally): Attempting to join lingering AprilTag thread...")
                apriltag_thread.join(timeout=2.0)
                if apriltag_thread.is_alive(): print("Main(finally): WARNING - Lingering AT thread did not terminate.")
                else: print("Main(finally): Lingering AT thread joined.")

        if not (game_instance and hasattr(game_instance, 'sandbox_ws_initialized')):

            if hasattr(sandbox_utils, 'g_ws_client_instance_util') and sandbox_utils.g_ws_client_instance_util is not None:
                print_once("main_final_cleanup_check_ws", "Main(finally): Game instance/loop might not have run fully. Ensuring WS client shutdown.")
                try:
                    sandbox_utils.shutdown_sandbox_websocket_client()
                except Exception as e_ws_final_shutdown:
                    print(f"Main(finally): Error during final WS client shutdown: {e_ws_final_shutdown}")

        if _ai_log_file_handle and not _ai_log_file_handle.closed: 
            print_once("main_final_cleanup_check_log", "Main(finally): Fallback AI log file close.")
            _ai_log_file_handle.write(f"--- AI LOG SESSION END @ {time.asctime()} (Main Finally Fallback) ---\n")
            _ai_log_file_handle.close()
            _ai_log_file_handle = None 

        if pygame.get_init(): 
            pygame.quit()
        print("Main: Application terminated.")