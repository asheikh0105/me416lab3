import json
import paho.mqtt.client as mqtt

MQTT_BROKER = "rasticvm.lan"
MQTT_PORT = 1883
MQTT_TOPIC = "rb/limo777/command"

def limo_command(v, w, client):
    msg = json.dumps({"v": v, "w": w})
    client.publish(MQTT_TOPIC, msg)
    print("Sent MQTT command:", msg)

def client_connect(address, port, topics):
    """
    Connect to an MQTT broker and subscribe to the given topics.
    Returns the MQTT client object with peek() capability.
    """

    client = mqtt.Client(client_id="myPythonClient")
    
    # Store last message
    client._last_message = None

    # on-connect callback (prints status)
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker.")
        else:
            print(f"Failed to connect to MQTT broker (rc={rc})")

    # on-message callback to store incoming messages
    def on_message(client, userdata, msg):
        client._last_message = msg.payload.decode('utf-8')
        
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect
    client.connect(address, port)

    # Start internal network loop
    client.loop_start()

    # Subscribe to topics (same as MATLAB for-loop)
    for topic in topics:
        client.subscribe(topic)
        print(f"Subscribed to topic: {topic}")
    
    # Add peek method to client
    def peek():
        return client._last_message
    
    client.peek = peek

    return client