# Copyright (c) 2025 Oguzhan Cagirir (OguzhanCOG), KCL Electronics Society
#
# Project: KCL RoboFootball System (FoAI Fork)
# File: KILL.py
# Description: Halt all robots on execution by sending a stop command via MQTT.
#
# Author: Oguzhan Cagirir (OguzhanCOG)
# Date: December 2, 2025
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
# --- Version: 1.1.0 (MQTT) ---

import time
from ws_client import init_global_ws_client, get_global_ws_client, shutdown_global_ws_client

ROBOT_IDS = [1, 2, 3, 4]

def main():
    print("Initializing MQTT client...")
    init_global_ws_client()
    time.sleep(1)

    client = get_global_ws_client()
    if not client or not client.check_if_actively_connected():
        print("ERROR: MQTT client not connected. Cannot send commands.")
        return

    print("Client connected. Sending STOP command (0,0) to all robots...")
    for robot_id in ROBOT_IDS:
        client.send_joystick_command(userid=robot_id, x=0, y=0)
        print(f"-> Sent stop command to robot ID {robot_id}")

    time.sleep(1)
    print("Shutting down MQTT client...")
    shutdown_global_ws_client()
    print("Killed.")

if __name__ == "__main__":
    main()