# Copyright (c) 2025 Oguzhan Cagirir (OguzhanCOG), KCL Electronics Society
#
# Project: KCL RoboFootball System (FoAI Fork)
# File: ws_client.py (MQTT EDITION)
# Description: MQTT client acting as a drop-in replacement for the original WebSocket client. Sends joystick commands directly to the broker.
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
# --- Version: 2.0.0 (MQTT) ---

import json
import time
import paho.mqtt.client as mqtt
import threading

# ==========================================
# CONFIGURATION
# ==========================================

MQTT_BROKER_IP = "46.224.81.207"
MQTT_PORT = 1883

# MAP SIMULATION IDs TO ESP32 MAC ADDRESSES HERE
# Use Arduino Serial Monitor (e.g., "robot/registry/246F28A1B2C3")
ROBOT_MAPPING = {
    1: "REPLACE_WITH_MAC_ADDRESS_FOR_ROBOT_1",  # e.g., "A1B2C3D4E5F6"
    2: "REPLACE_WITH_MAC_ADDRESS_FOR_ROBOT_2",
    3: "REPLACE_WITH_MAC_ADDRESS_FOR_ROBOT_3",
    4: "REPLACE_WITH_MAC_ADDRESS_FOR_ROBOT_4"
}

# ==========================================

_client_instance = None
_printed_messages = set()

def print_once(key, msg):
    if key not in _printed_messages:
        print(msg)
        _printed_messages.add(key)

class MQTTGameClient:
    def __init__(self, broker_ip, port):
        self.broker_ip = broker_ip
        self.port = port
        self.client = mqtt.Client(client_id="Sim_Controller", protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.connected = False
        self.running = False

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"MQTT_CLIENT: Connected to Broker {self.broker_ip}")
        else:
            print(f"MQTT_CLIENT: Connection failed with code {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        print("MQTT_CLIENT: Disconnected from Broker")

    def start(self):
        if self.running: return
        print(f"MQTT_CLIENT: Connecting to {self.broker_ip}:{self.port}...")
        try:
            self.client.connect(self.broker_ip, self.port, 60)
            self.client.loop_start()
            self.running = True
        except Exception as e:
            print(f"MQTT_CLIENT: Failed to start: {e}")

    def stop(self):
        if not self.running: return
        print("MQTT_CLIENT: Stopping...")
        self.client.loop_stop()
        self.client.disconnect()
        self.running = False
        self.connected = False

    def check_if_actively_connected(self):
        return self.running and self.connected

    def send_joystick_command(self, userid: int, x: float, y: float):
        """
        Sends the {x, y} payload to the specific robot's control topic.
        """
        if not self.check_if_actively_connected():
            return

        if userid not in ROBOT_MAPPING:
            print_once(f"missing_mac_{userid}", f"MQTT_CLIENT: No MAC address mapped for Robot ID {userid}!")
            return

        mac_address = ROBOT_MAPPING[userid]
        topic = f"robot/control/{mac_address}"
        
        payload = json.dumps({
            "x": round(x, 2),
            "y": round(y, 2)
        })

        try:
            self.client.publish(topic, payload, qos=0)
            # print(f"Sent to {topic}: {payload}") # Uncomment for verbose debugging
        except Exception as e:
            print(f"MQTT_CLIENT: Error publishing: {e}")

def init_global_ws_client():
    global _client_instance
    if _client_instance is None:
        _client_instance = MQTTGameClient(MQTT_BROKER_IP, MQTT_PORT)
        _client_instance.start()
    else:
        print("MQTT_CLIENT: Already initialized.")

def get_global_ws_client():
    global _client_instance
    return _client_instance

def shutdown_global_ws_client():
    global _client_instance
    if _client_instance:
        _client_instance.stop()
        _client_instance = None

if __name__ == '__main__':
    print("Testing MQTT Client...")
    init_global_ws_client()
    time.sleep(1)
    
    cl = get_global_ws_client()
    if cl.check_if_actively_connected():
        print("Connected! Sending test command to Robot 1...")
        # Simulates "Forward"
        cl.send_joystick_command(1, 0.0, 1.0) 
        time.sleep(1)
        cl.send_joystick_command(1, 0.0, 0.0)
    else:
        print("Failed to connect.")
    
    shutdown_global_ws_client()