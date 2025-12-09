#!/usr/bin/env python3

import json
import socket
import time
import paho.mqtt.client as mqtt
import signal
import sys

MQTT_BROKER = "rasticvm.lan"
MQTT_PORT = 1883
MQTT_TOPIC = "rb/limo809/command"

# Limo motor TCP server
LIMO_IP = "192.168.1.172"
LIMO_PORT = 12345

# OPTIONAL delay between messages (seconds)
SEND_DELAY = 0.05   # 50 ms – change this as needed

# --- CLEAN EXIT HANDLER ------------------------------------------------------

def clean_exit(*args):
    print("\nShutting down cleanly...")

    try:
        if mq.is_connected():
            mq.disconnect()
            print("Disconnected MQTT.")
    except:
        pass

    try:
        sock.close()
        print("Closed TCP socket.")
    except:
        pass

    sys.exit(0)

# Register Ctrl-C / SIGINT handler
signal.signal(signal.SIGINT, clean_exit)

# -----------------------------------------------------------------------------


# Create TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print("Connecting to LIMO...")
sock.connect((LIMO_IP, LIMO_PORT))
print("Connected to LIMO.")


def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker.")
    client.subscribe(MQTT_TOPIC)
    print(f"Subscribed to {MQTT_TOPIC}")


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())

        v = data["v"]
        w = data["w"]

        cmd = f"{v},{w}"

        sock.send(cmd.encode())

        print(f"Sent to Limo: {cmd}")

        # Optional rate limiting
        time.sleep(SEND_DELAY)

    except Exception as e:
        print("Error:", e)


# MQTT setup
mq = mqtt.Client()
mq.on_connect = on_connect
mq.on_message = on_message

print("Connecting to MQTT...")
mq.connect(MQTT_BROKER, MQTT_PORT, 60)
print("Bridge fully running. Ctrl-C to exit.\n")

mq.loop_forever()