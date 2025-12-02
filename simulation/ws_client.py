# Copyright (c) 2025 Oguzhan Cagirir (OguzhanCOG), KCL Electronics Society
#
# Project: KCL FoAI RoboFootball System
# File: ws_client.py
# Description: Asynchronous WebSocket client for sending joystick commands and managing connection to a remote server.
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

import asyncio
import websockets
import json
import threading
import time
import traceback

_ws_client_printed_messages = set()
def print_once_ws(message_key, message_content):
    global _ws_client_printed_messages
    if message_key not in _ws_client_printed_messages:
        print(message_content)
        _ws_client_printed_messages.add(message_key)

class WebSocketGameClient:
    def __init__(self, uri):
        self.uri = uri
        self.loop = None
        self.thread = None
        self.running = False
        self.is_connected = False
        self.command_queue = asyncio.Queue(maxsize=100)
        print(f"WS_CLIENT_INIT: Instance created for {self.uri}. Running: {self.running}, Connected: {self.is_connected}")

    async def _connect_and_send_loop(self):
        print("WS_CLIENT_THREAD: _connect_and_send_loop started.")
        while self.running:
            connection = None
            self.is_connected = False

            try:
                print(f"WS_CLIENT_THREAD: Attempting to connect to {self.uri}...")
                connection = await asyncio.wait_for(websockets.connect(self.uri), timeout=5.0)
                print(f"WS_CLIENT_THREAD: Successfully connected to WebSocket: {self.uri}")
                self.is_connected = True

                while self.running and self.is_connected:
                    try:
                        command_payload = await asyncio.wait_for(self.command_queue.get(), timeout=0.1)
                        if command_payload is None:
                            print("WS_CLIENT_THREAD: Got None shutdown signal from queue.")
                            self.is_connected = False
                            if not self.running:
                                self.command_queue.task_done()
                                print("WS_CLIENT_THREAD: Shutdown signal processed, exiting inner send loop.")
                                return
                            else:
                                print("WS_CLIENT_THREAD: Got None from queue but still running? Breaking inner loop.")
                                self.command_queue.task_done()
                                break
                        if command_payload:
                            await connection.send(json.dumps(command_payload))
                            self.command_queue.task_done()
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosedOK:
                        print("WS_CLIENT_THREAD: WebSocket connection closed normally (OK).")
                        self.is_connected = False; break
                    except websockets.exceptions.ConnectionClosedError as e:
                        print(f"WS_CLIENT_THREAD: WebSocket connection closed with error: {e}.")
                        self.is_connected = False; break
                    except Exception as e_inner:
                        print(f"WS_CLIENT_THREAD: Error in WebSocket send_loop (inner): {e_inner}")

                        await asyncio.sleep(0.1)
                if not self.running: print("WS_CLIENT_THREAD: self.running is False, breaking outer connection loop."); break
            except asyncio.TimeoutError: print(f"WS_CLIENT_THREAD: WebSocket connection to {self.uri} timed out. Will retry.")
            except (websockets.exceptions.InvalidURI, websockets.exceptions.InvalidHandshake, ConnectionRefusedError, OSError) as e_conn_protocol:
                print(f"WS_CLIENT_THREAD: WebSocket connection/protocol error (outer): {e_conn_protocol}. Will retry.")
            except Exception as e_outer:
                print(f"WS_CLIENT_THREAD: Unexpected WebSocket error in _connect_and_send_loop (outer): {e_outer}"); traceback.print_exc() 
            finally:

                self.is_connected = False
                if connection:
                    try:

                        await connection.close()
                    except Exception: pass 
                connection = None
            if not self.running: print("WS_CLIENT_THREAD: self.running is False after outer 'finally', exiting _connect_and_send_loop."); break
            if self.running:

                try:
                    for _ in range(50):
                        if not self.running: break
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError: print("WS_CLIENT_THREAD: Sleep interrupted for shutdown."); break
        print("WS_CLIENT_THREAD: _connect_and_send_loop finished."); self.is_connected = False

    def _run_event_loop(self):
        print("WS_CLIENT: _run_event_loop creating new event loop.")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        print(f"WS_CLIENT: Event loop created. self.running is {self.running}. Starting _connect_and_send_loop.")
        try:
            self.loop.run_until_complete(self._connect_and_send_loop())
        except Exception as e: print(f"WS_CLIENT: Error running WebSocket event loop: {e}"); traceback.print_exc()
        finally:
            print("WS_CLIENT: _run_event_loop 'finally'. Cleaning queue...")
            while not self.command_queue.empty():
                try: self.command_queue.get_nowait(); self.command_queue.task_done()
                except: break 
            if self.loop and self.loop.is_running():
                print("WS_CLIENT: Stopping asyncio event loop from _run_event_loop finally...")
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop and not self.loop.is_closed():
                print("WS_CLIENT: Closing asyncio event loop...")

                all_tasks = asyncio.all_tasks(loop=self.loop)
                if all_tasks:
                    self.loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))
                self.loop.close()
                print("WS_CLIENT: Asyncio event loop closed.")
            else: print("WS_CLIENT: Asyncio event loop None, not running, or already closed in _run_event_loop finally.")
            print("WS_CLIENT: WebSocket event loop fully stopped from _run_event_loop.")
        self.running = False; self.is_connected = False

    def start(self):
        if self.thread and self.thread.is_alive():
            print("WS_CLIENT: Client thread already alive and running.")
            if not self.running: 
                self.running = True 
                print("WS_CLIENT: (Thread was alive but running flag was false, reset to True)")
            return

        print("WS_CLIENT: Starting new client thread...")
        self.running = True 
        self.is_connected = False 
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        print("WS_CLIENT: Client thread started.")

    def stop(self):
        print("WS_CLIENT: Attempting to stop client...")
        if not self.running and not (self.thread and self.thread.is_alive()):
            print("WS_CLIENT: Client already stopped or not started effectively.")
            return

        self.running = False 
        self.is_connected = False

        if self.loop and self.loop.is_running(): 
            print("WS_CLIENT: Queuing shutdown signal (None) to command_queue...")
            future = asyncio.run_coroutine_threadsafe(self.command_queue.put(None), self.loop)
            try: future.result(timeout=1.0); print("WS_CLIENT: Shutdown signal queued successfully.")
            except TimeoutError: print("WS_CLIENT: Timeout putting shutdown signal on queue.")
            except Exception as e: print(f"WS_CLIENT: Exception putting shutdown signal: {e}")
        else: print("WS_CLIENT: Event loop not available/running for queuing shutdown signal.")

        if self.thread and self.thread.is_alive():
            print(f"WS_CLIENT: Joining client thread (timeout 7s)...")
            self.thread.join(timeout=7.0)
            if self.thread.is_alive(): print("WS_CLIENT: Warning: Client thread did not stop gracefully.")
            else: print("WS_CLIENT: Client thread joined.")
        else: print("WS_CLIENT: Client thread was not alive to join.")

        self.loop = None
        print("WS_CLIENT: Client stop process complete.")

    def check_if_actively_connected(self):
        return self.running and self.is_connected

    def send_joystick_command(self, userid: int, x: float, y: float):
        if not self.check_if_actively_connected():

            return
        payload = {"type": "joystick", "userid": userid, "joystick": {'x': round(x,2), 'y': round(y,2)}}
        try: self.command_queue.put_nowait(payload)
        except queue.Full: 
             print_once_ws("ws_q_full", "WS_CLIENT (send_cmd): Command queue full. Cmd dropped.")
        except Exception as e: print_once_ws(f"ws_q_put_err_{e}", f"WS_CLIENT (send_cmd): Error on queue put: {e}")

WS_CLIENT_URI = "ws://89.117.63.5:8000/ws/mobilecontrol"
_ws_game_client_instance = None 

def init_global_ws_client():
    global _ws_game_client_instance
    if _ws_game_client_instance is None or not _ws_game_client_instance.running:
        if _ws_game_client_instance and not _ws_game_client_instance.running:
            print("WS_CLIENT_GLOBAL: Previous client instance was stopped. Ensuring full cleanup before re-initializing.")
            _ws_game_client_instance.stop() 
            _ws_game_client_instance = None
        print(f"WS_CLIENT_GLOBAL: Initializing WebSocket client for URI: {WS_CLIENT_URI}")
        _ws_game_client_instance = WebSocketGameClient(WS_CLIENT_URI)
        _ws_game_client_instance.start()
    else:
        print("WS_CLIENT_GLOBAL: WebSocket client already initialized and appears to be running.")

def get_global_ws_client(): 
    global _ws_game_client_instance
    return _ws_game_client_instance

def shutdown_global_ws_client():
    global _ws_game_client_instance
    if _ws_game_client_instance:
        print("WS_CLIENT_GLOBAL: Shutting down WebSocket client...")
        _ws_game_client_instance.stop()
        _ws_game_client_instance = None
        print("WS_CLIENT_GLOBAL: Global WebSocket client instance shut down and cleared.")
    else:
        print("WS_CLIENT_GLOBAL: No active WebSocket client to shut down.")

if __name__ == '__main__': 
    print("Starting WebSocket client test (ws_client.py)...")
    init_global_ws_client()
    client = get_global_ws_client()

    start_time = time.time(); connection_wait_time = 7
    print(f"Waiting up to {connection_wait_time}s for initial connection attempt...")
    while time.time() - start_time < connection_wait_time:
        if client and client.check_if_actively_connected(): print("TEST: Client connected!"); break
        time.sleep(0.5)
    else:
        if client: print(f"TEST: Client did not connect. Status: running={client.running}, connected={client.is_connected}")
        else: print(f"TEST: Client is None after {connection_wait_time}s.")

    if client and client.check_if_actively_connected():
        print("TEST: Sending test joystick command..."); client.send_joystick_command(userid=1, x=0.51, y=-0.26)
        time.sleep(2); print("TEST: Sending another test command..."); client.send_joystick_command(userid=1, x=0.1, y=0.1)
        time.sleep(2)
    else: print("TEST: WebSocket client not actively connected, test command not sent.")
    print("TEST: Shutting down WebSocket client..."); shutdown_global_ws_client()
    print("TEST: Test finished (ws_client.py).")