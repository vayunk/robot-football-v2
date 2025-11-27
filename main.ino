#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <math.h>

// -------- WiFi credentials --------
const char *ssid = "12345678";
const char *password = "abcdefghi@";

// -------- MQTT broker (Mosquitto over WebSockets) --------
const char *mqtt_server = "46.224.81.207";
const uint16_t mqtt_port = 1883;

WiFiClient espClient;
PubSubClient client(espClient);

// -------- Motor pins --------
#define M1_FORWARD_PIN 17 //topleft
#define M1_BACKWARD_PIN 5
#define M2_FORWARD_PIN 18 //topright
#define M2_BACKWARD_PIN 19
#define M3_FORWARD_PIN 4 //bottom
#define M3_BACKWARD_PIN 16

int currentGlobalSpeed = 200;

String macClean;   // MAC address without colons
String registryTopic;
String controlTopic;

// -------- Movement functions (same as before) --------
void moveXY(float x, float y);
void setMotor(int forwardPin, int backwardPin, int speed);
void rotateClockwise();
void rotateCounterClockwise();
void stop();

// -------- MQTT callback --------
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String data;
  for (unsigned int i = 0; i < length; i++) {
    data += (char)payload[i];
  }
  data.trim();

  Serial.print("Message on [");
  Serial.print(topic);
  Serial.print("]: ");
  Serial.println(data);

  JsonDocument doc;
  DeserializationError error = deserializeJson(doc, data);
  if (error) {
    Serial.print(F("deserializeJson() failed: "));
    Serial.println(error.f_str());
    return;
  }

  float x = doc["x"];
  float y = doc["y"];

  if (x == 99 && y == 99) {
    Serial.println("Rotating clockwise");
    rotateClockwise();
    return;
  }
  if (x == -99 && y == -99) {
    Serial.println("Rotating counter-clockwise");
    rotateCounterClockwise();
    return;
  }

  if (y > 0) {
    y = -y;
  } else {
    y = abs(y);
  }

  moveXY(x, y);
}

// -------- MQTT reconnect with LWT --------
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");

    String clientId = "ESP32-" + macClean;

    // Prepare LWT (if this robot disconnects uncleanly)
    String willMessage = "{\"status\":\"offline\"}";

    if (client.connect(clientId.c_str(),
                       NULL, NULL,                       // no username/pass
                       registryTopic.c_str(),            // LWT topic
                       1,                                // QoS
                       true,                             // retained
                       willMessage.c_str())) {           // LWT payload

      Serial.println("connected");

      // Publish ONLINE status (retained)
      String onlineMsg = "{\"status\":\"online\"}";
      client.publish(registryTopic.c_str(), onlineMsg.c_str(), true);

      // Subscribe to control topic
      client.subscribe(controlTopic.c_str());

      Serial.print("Subscribed to: ");
      Serial.println(controlTopic);
      Serial.print("Registered on: ");
      Serial.println(registryTopic);

    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5s");
      delay(5000);
    }
  }
}

// -------- Setup --------
void setup() {
  Serial.begin(115200);

  // Motor pins
  pinMode(M1_FORWARD_PIN, OUTPUT);
  pinMode(M1_BACKWARD_PIN, OUTPUT);
  pinMode(M2_FORWARD_PIN, OUTPUT);
  pinMode(M2_BACKWARD_PIN, OUTPUT);
  pinMode(M3_FORWARD_PIN, OUTPUT);
  pinMode(M3_BACKWARD_PIN, OUTPUT);

  // WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected. IP: " + WiFi.localIP().toString());

  // Clean MAC (remove colons)
  macClean = WiFi.macAddress();
  macClean.replace(":", "");

  // Topics for this robot
  registryTopic = "robot/registry/" + macClean;
  controlTopic = "robot/control/" + macClean;

  // MQTT setup
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(mqttCallback);
}

// -------- Loop --------
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}

// -------- Movement code (unchanged) --------
void rotateClockwise() {
  int speed = currentGlobalSpeed;
  setMotor(M1_FORWARD_PIN, M1_BACKWARD_PIN, speed);
  setMotor(M2_FORWARD_PIN, M2_BACKWARD_PIN, speed);
  setMotor(M3_FORWARD_PIN, M3_BACKWARD_PIN, speed);
}

void rotateCounterClockwise() {
  int speed = currentGlobalSpeed;
  setMotor(M1_FORWARD_PIN, M1_BACKWARD_PIN, -speed);
  setMotor(M2_FORWARD_PIN, M2_BACKWARD_PIN, -speed);
  setMotor(M3_FORWARD_PIN, M3_BACKWARD_PIN, -speed);
}

void moveXY(float x, float y) {
    float mag = sqrt(x*x + y*y);
    if (mag > 1.0) { x /= mag; y /= mag; }

    // Wheel speeds with correct motor layout
    float wA = 0.5*x - 0.866*y;   // Motor A (top-left, 17/5)
    float wB = 0.5*x + 0.866*y;   // Motor B (top-right, 18/19)
    float wC = -x;                 // Motor C (bottom, 4/16)

    int mA = int(wA * currentGlobalSpeed);
    int mB = int(wB * currentGlobalSpeed);
    int mC = int(wC * currentGlobalSpeed);

    Serial.printf("moveXY x: %.2f y: %.2f | A=%d B=%d C=%d\n", x, y, mA, mB, mC);

    setMotor(M1_FORWARD_PIN, M1_BACKWARD_PIN, mA);   // Motor A (top-left)
    setMotor(M2_FORWARD_PIN, M2_BACKWARD_PIN, mB);  // Motor B (top-right)
    setMotor(M3_FORWARD_PIN, M3_BACKWARD_PIN, mC);   // Motor C (bottom)
}


void setMotor(int forwardPin, int backwardPin, int speed) {
  if (speed > 0) {
    analogWrite(forwardPin, abs(speed));
    analogWrite(backwardPin, 0);
  } else if (speed < 0) {
    analogWrite(forwardPin, 0);
    analogWrite(backwardPin, abs(speed));
  } else {
    analogWrite(forwardPin, 0);
    analogWrite(backwardPin, 0);
  }
}
