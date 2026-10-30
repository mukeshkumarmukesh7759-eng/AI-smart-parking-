# ============================================================
#  Smart Parking System - Serial Bridge
#  Reads live JSON data from Arduino over USB Serial and
#  feeds it to the dashboard / logs it to CSV.
#
#  Usage:
#    python src/serial_bridge.py --port COM3 --baud 9600
# ============================================================

import serial
import json
import csv
import os
import time
import argparse
import threading
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "live_readings.csv")

# ── CSV logger ───────────────────────────────────────────────
def init_csv(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "slot_id", "distance_cm", "occupied"
            ])
    return path


def log_to_csv(path: str, readings: list, timestamp: str):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        for r in readings:
            writer.writerow([
                timestamp, r["id"], r["dist"], r["occupied"]
            ])


# ── Display in terminal ───────────────────────────────────────
def display(data: dict, timestamp: str):
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 50)
    print(f"  Smart Parking System  |  {timestamp}")
    print("=" * 50)
    print(f"  Occupied : {data['occupied']} / {data['total']}")
    print(f"  Vacant   : {data['vacant']}  / {data['total']}")
    print("-" * 50)
    for s in data["slots"]:
        bar   = "[OCC]" if s["occupied"] else "[VAC]"
        print(f"  Slot {s['id']:>2}  {bar}  {s['dist']:6.1f} cm")
    print("=" * 50)


# ── Main serial read loop ─────────────────────────────────────
def read_serial(port: str, baud: int, log: bool = True):
    csv_path = init_csv(LOG_PATH) if log else None

    print(f"[*] Connecting to {port} at {baud} baud ...")
    try:
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)   # wait for Arduino reset
        print(f"[+] Connected! Reading data ...\n")
    except serial.SerialException as e:
        print(f"[ERROR] Could not open {port}: {e}")
        print("\nAvailable ports:")
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}  -  {p.description}")
        return

    try:
        while True:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw or not raw.startswith("{"):
                continue
            try:
                data = json.loads(raw)
                if "slots" not in data:
                    continue
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                display(data, ts)
                if log and csv_path:
                    log_to_csv(csv_path, data["slots"], ts)
            except json.JSONDecodeError:
                pass
    except KeyboardInterrupt:
        print("\n[*] Serial bridge stopped.")
    finally:
        ser.close()


# ── Simulation mode (no hardware needed) ─────────────────────
def simulate_mode(log: bool = True):
    """Run without Arduino using the software sensor simulator."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from sensor_simulator import ParkingLotSimulator

    csv_path = init_csv(LOG_PATH) if log else None
    sim = ParkingLotSimulator()
    print("[*] Running in SIMULATION mode (no Arduino required)")
    print("    Press Ctrl+C to stop\n")

    try:
        while True:
            readings = sim.read_all_slots()
            summary  = sim.summary(readings)
            ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Convert to same format as Arduino JSON
            data = {
                "slots"   : [{"id": r["slot_id"],
                               "dist": r["distance_cm"],
                               "occupied": r["occupied"]} for r in readings],
                "total"   : summary["total_slots"],
                "occupied": summary["occupied"],
                "vacant"  : summary["vacant"],
            }
            display(data, ts)
            if log and csv_path:
                log_to_csv(csv_path, data["slots"], ts)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n[*] Simulation stopped.")


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Parking - Serial Bridge / Simulator"
    )
    parser.add_argument("--port",     type=str, default=None,
                        help="COM port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud",     type=int, default=9600)
    parser.add_argument("--no-log",   action="store_true",
                        help="Disable CSV logging")
    parser.add_argument("--simulate", action="store_true",
                        help="Run in simulation mode (no Arduino)")
    args = parser.parse_args()

    if args.simulate or args.port is None:
        simulate_mode(log=not args.no_log)
    else:
        read_serial(args.port, args.baud, log=not args.no_log)
