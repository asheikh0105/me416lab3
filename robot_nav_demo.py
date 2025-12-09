import paho.mqtt.client as mqtt
from enum import Enum
import json
import time
from typing import Tuple, Optional

class RobotState(Enum):
    """Robot run states as defined in the requirements"""
    WAIT = "WAIT"
    GO = "GO"
    STOP = "STOP"
    HALT = "HALT"

class RobotStateMachine:
    def __init__(self, robot_id: str, mqtt_broker: str = "localhost"):
        self.robot_id = robot_id
        self.current_state = RobotState.WAIT
        self.goal_position: Optional[Tuple[float, float]] = None
        self.mqtt_client = mqtt.Client()
        self.mqtt_broker = mqtt_broker
        
        # Setup MQTT callbacks
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
        # Flag to track if we should exit
        self.should_exit = False
        
    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        print(f"Connected to MQTT broker with result code {rc}")
        
        # Subscribe to command and goal topics
        cmd_topic = f"cmd/limo{self.robot_id}"
        goal_topic = f"goal/limo{self.robot_id}"
        
        client.subscribe(cmd_topic)
        client.subscribe(goal_topic)
        print(f"Subscribed to {cmd_topic} and {goal_topic}")
        
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        if topic.startswith("cmd/"):
            self.handle_command(payload)
        elif topic.startswith("goal/"):
            self.handle_goal(payload)
    
    def handle_command(self, command: str):
        """Handle run state commands"""
        print(f"Received command: {command}")
        
        try:
            new_state = RobotState(command)
            self.transition_to(new_state)
        except ValueError:
            print(f"Unknown command: {command}")
    
    def handle_goal(self, goal_json: str):
        """Handle goal position updates"""
        # Only update goal if in WAIT state
        if self.current_state != RobotState.WAIT:
            print(f"Ignoring goal update - not in WAIT state (current: {self.current_state.value})")
            return
        
        try:
            goal_data = json.loads(goal_json)
            x, y = goal_data["goal"]
            self.goal_position = (x, y)
            print(f"Goal updated to: ({x}, {y})")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing goal message: {e}")
    
    def transition_to(self, new_state: RobotState):
        """Handle state transitions with appropriate actions"""
        old_state = self.current_state
        self.current_state = new_state
        
        print(f"State transition: {old_state.value} -> {new_state.value}")
        
        # Execute state-specific actions
        if new_state == RobotState.WAIT:
            self.on_wait()
        elif new_state == RobotState.GO:
            self.on_go()
        elif new_state == RobotState.STOP:
            self.on_stop()
        elif new_state == RobotState.HALT:
            self.on_halt()
    
    def on_wait(self):
        """Actions when entering WAIT state"""
        print("Entering WAIT state - ready to receive goal")
        self.stop_robot()
    
    def on_go(self):
        """Actions when entering GO state"""
        print("Entering GO state - starting navigation")
        if self.goal_position:
            print(f"Navigating to goal: {self.goal_position}")
            # Start your navigation/path planning here
        else:
            print("Warning: No goal set!")
    
    def on_stop(self):
        """Actions when entering STOP state - must stop within 100ms"""
        print("Entering STOP state - stopping robot (100ms deadline)")
        self.stop_robot()
        # Robot can resume when GO is received
    
    def on_halt(self):
        """Actions when entering HALT state - must stop within 100ms and exit"""
        print("Entering HALT state - stopping robot and preparing to exit")
        self.stop_robot()
        self.should_exit = True
        # Graceful shutdown - cleanup resources
    
    def stop_robot(self):
        """Send stop command to robot motors"""
        # TODO: Implement your robot's stop command
        # Example: self.robot.set_velocity(0, 0)
        print("Robot motors stopped")
    
    def run(self):
        """Main control loop"""
        # Connect to MQTT broker
        self.mqtt_client.connect(self.mqtt_broker, 1883, 60)
        self.mqtt_client.loop_start()
        
        print("Robot state machine running...")
        
        try:
            while not self.should_exit:
                # Main control logic based on current state
                if self.current_state == RobotState.GO:
                    self.navigate_step()
                elif self.current_state == RobotState.WAIT:
                    # Just idle
                    pass
                elif self.current_state == RobotState.STOP:
                    # Stay stopped
                    pass
                
                time.sleep(0.01)  # 10ms loop rate
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        finally:
            self.cleanup()
    
    def navigate_step(self):
        """Execute one step of navigation/obstacle avoidance"""
        if not self.goal_position:
            return
        
        # TODO: Implement your navigation logic here
        # 1. Get current position from mocap
        # 2. Get positions of other robots (from mocap)
        # 3. Calculate path/velocity avoiding obstacles
        # 4. Send velocity commands to robot
        
        pass
    
    def cleanup(self):
        """Cleanup resources before exit"""
        print("Cleaning up...")
        self.stop_robot()
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        print("Exited gracefully")


# Example usage
if __name__ == "__main__":
    # Replace with your robot ID (e.g., "777", "805")
    ROBOT_ID = "777"
    MQTT_BROKER = "rasticvm.lan" 
    
    robot = RobotStateMachine(ROBOT_ID, MQTT_BROKER)
    robot.run()