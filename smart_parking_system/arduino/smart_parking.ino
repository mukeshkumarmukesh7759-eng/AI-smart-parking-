// ============================================================
//  Smart Parking System - Arduino Code
//  Hardware: Arduino Uno + HC-SR04 Ultrasonic Sensors
//            + LEDs (Red = Occupied, Green = Vacant)
//            + Optional Servo (Automated Gate)
//
//  Wiring per Slot (repeat for each slot):
//    HC-SR04 TRIG  -> Digital Pin (TRIG_PINS[i])
//    HC-SR04 ECHO  -> Digital Pin (ECHO_PINS[i])
//    Green LED     -> Digital Pin (GREEN_PINS[i])
//    Red   LED     -> Digital Pin (RED_PINS[i])
//  Gate Servo      -> Digital Pin 2
// ============================================================

#include <Servo.h>

// ── Number of parking slots ───────────────────────────────
#define NUM_SLOTS 5         // adjust to match your hardware

// ── Sensor pins ───────────────────────────────────────────
const int TRIG_PINS[NUM_SLOTS]  = {22, 26, 30, 34, 38};
const int ECHO_PINS[NUM_SLOTS]  = {23, 27, 31, 35, 39};

// ── LED indicator pins ────────────────────────────────────
const int GREEN_PINS[NUM_SLOTS] = {24, 28, 32, 36, 40};
const int RED_PINS[NUM_SLOTS]   = {25, 29, 33, 37, 41};

// ── Gate servo ────────────────────────────────────────────
const int SERVO_PIN  = 2;
const int GATE_OPEN  = 90;   // degrees
const int GATE_CLOSE = 0;

// ── Occupancy threshold ───────────────────────────────────
const float THRESHOLD_CM = 10.0;  // < 10 cm → occupied

// ── Timing ───────────────────────────────────────────────
const unsigned long READ_INTERVAL = 1000;  // ms between reads
unsigned long lastRead = 0;

Servo gateServo;

// ─────────────────────────────────────────────────────────
//  Measure distance from HC-SR04
// ─────────────────────────────────────────────────────────
float measureDistance(int trigPin, int echoPin) {
    // Send 10µs pulse
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    // Read echo duration (timeout = 30ms)
    long duration = pulseIn(echoPin, HIGH, 30000);
    if (duration == 0) return 999.0;  // timeout → no object

    // Distance = duration * speed_of_sound / 2
    return (duration * 0.0343) / 2.0;
}

// ─────────────────────────────────────────────────────────
//  Update LED indicators
// ─────────────────────────────────────────────────────────
void updateLED(int slot, bool occupied) {
    digitalWrite(GREEN_PINS[slot], occupied ? LOW  : HIGH);
    digitalWrite(RED_PINS[slot],   occupied ? HIGH : LOW);
}

// ─────────────────────────────────────────────────────────
//  Control gate based on total availability
// ─────────────────────────────────────────────────────────
void controlGate(int vacantCount) {
    if (vacantCount > 0) {
        gateServo.write(GATE_OPEN);   // open gate
    } else {
        gateServo.write(GATE_CLOSE);  // lot full — close gate
    }
}

// ─────────────────────────────────────────────────────────
//  Send JSON over Serial for Python to read
//  Format: {"slots":[{"id":1,"dist":8.2,"occupied":1}, ...],
//           "total":5,"occupied":3,"vacant":2}
// ─────────────────────────────────────────────────────────
void sendSerialJSON(float distances[], bool occupied[],
                    int totalOcc, int totalVac) {
    Serial.print("{\"slots\":[");
    for (int i = 0; i < NUM_SLOTS; i++) {
        Serial.print("{\"id\":");
        Serial.print(i + 1);
        Serial.print(",\"dist\":");
        Serial.print(distances[i], 1);
        Serial.print(",\"occupied\":");
        Serial.print(occupied[i] ? 1 : 0);
        Serial.print("}");
        if (i < NUM_SLOTS - 1) Serial.print(",");
    }
    Serial.print("],\"total\":");
    Serial.print(NUM_SLOTS);
    Serial.print(",\"occupied\":");
    Serial.print(totalOcc);
    Serial.print(",\"vacant\":");
    Serial.print(totalVac);
    Serial.println("}");
}

// ─────────────────────────────────────────────────────────
//  Setup
// ─────────────────────────────────────────────────────────
void setup() {
    Serial.begin(9600);

    for (int i = 0; i < NUM_SLOTS; i++) {
        pinMode(TRIG_PINS[i],  OUTPUT);
        pinMode(ECHO_PINS[i],  INPUT);
        pinMode(GREEN_PINS[i], OUTPUT);
        pinMode(RED_PINS[i],   OUTPUT);

        // Start all slots as Vacant (green on)
        digitalWrite(GREEN_PINS[i], HIGH);
        digitalWrite(RED_PINS[i],   LOW);
    }

    gateServo.attach(SERVO_PIN);
    gateServo.write(GATE_OPEN);

    Serial.println("{\"status\":\"Smart Parking System Ready\"}");
}

// ─────────────────────────────────────────────────────────
//  Main Loop
// ─────────────────────────────────────────────────────────
void loop() {
    unsigned long now = millis();
    if (now - lastRead < READ_INTERVAL) return;
    lastRead = now;

    float distances[NUM_SLOTS];
    bool  occupied[NUM_SLOTS];
    int   totalOcc = 0, totalVac = 0;

    for (int i = 0; i < NUM_SLOTS; i++) {
        distances[i] = measureDistance(TRIG_PINS[i], ECHO_PINS[i]);
        occupied[i]  = (distances[i] < THRESHOLD_CM);

        updateLED(i, occupied[i]);

        if (occupied[i]) totalOcc++;
        else             totalVac++;
    }

    controlGate(totalVac);
    sendSerialJSON(distances, occupied, totalOcc, totalVac);
}
