# ============================================================
#  Smart Parking System – Sensor Simulator
#  Simulates live ultrasonic sensor readings for the dashboard.
# ============================================================

import numpy as np
import time
from datetime import datetime

NUM_SLOTS   = 10
THRESHOLD   = 50.0      # cm – below this = occupied

np.random.seed(None)    # fresh random on each run


def get_single_reading(slot_id: int,
                        prev_occupied: bool = False) -> dict:
    """
    Simulate one ultrasonic sensor reading for a given slot.
    Uses hysteresis: if slot was occupied, 70% chance to stay occupied.
    """
    if prev_occupied:
        prob = 0.70
    else:
        # Use hour-based probability
        hour = datetime.now().hour
        if 9 <= hour <= 17:
            prob = 0.75
        elif 7 <= hour <= 9 or 17 <= hour <= 19:
            prob = 0.55
        elif 19 <= hour <= 22:
            prob = 0.35
        else:
            prob = 0.10

    occupied = np.random.rand() < prob
    distance = (np.random.uniform(5, 30) if occupied
                else np.random.uniform(80, 200))

    return {
        "slot_id"    : slot_id,
        "distance_cm": round(float(distance), 2),
        "occupied"   : int(occupied),
        "status"     : "Occupied" if occupied else "Vacant",
        "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


class ParkingLotSimulator:
    """
    Maintains state for all slots and produces periodic updates.
    """

    def __init__(self, num_slots: int = NUM_SLOTS):
        self.num_slots = num_slots
        self._state    = [False] * (num_slots + 1)   # index 0 unused

    def read_all_slots(self) -> list:
        """Return a fresh reading for every slot."""
        readings = []
        for s in range(1, self.num_slots + 1):
            r = get_single_reading(s, self._state[s])
            self._state[s] = bool(r["occupied"])
            readings.append(r)
        return readings

    def summary(self, readings: list) -> dict:
        total    = len(readings)
        occupied = sum(r["occupied"] for r in readings)
        vacant   = total - occupied
        return {
            "total_slots"  : total,
            "occupied"     : occupied,
            "vacant"       : vacant,
            "occupancy_pct": round(occupied / total * 100, 1),
        }


# ── Quick CLI demo ────────────────────────────────────────────
if __name__ == "__main__":
    sim = ParkingLotSimulator()
    print("Live Parking Sensor Simulation  (Ctrl+C to stop)\n")
    try:
        while True:
            readings = sim.read_all_slots()
            summ     = sim.summary(readings)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}]  "
                  f"Occupied: {summ['occupied']}/{summ['total_slots']}  "
                  f"({summ['occupancy_pct']}%)")
            for r in readings:
                bar = "█" if r["occupied"] else "░"
                print(f"  Slot {r['slot_id']:>2}  {bar}  {r['status']:8s}  "
                      f"{r['distance_cm']:6.1f} cm")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
