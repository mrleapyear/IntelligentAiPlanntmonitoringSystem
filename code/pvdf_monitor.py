"""
Real-time PVDF Plant Health Monitor
Supports:
- Simulation mode (no ESP32)
- ESP32 serial input mode
"""

import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import os
import threading
import queue
import argparse

# ================================
# Real-Time Monitor Class
# ================================
class RealTimePVDFMonitor:

    def __init__(self, port=None, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False

        # Load ML model
        self.model, self.scaler, self.loaded = self.load_model()

        # Data buffers
        self.data_history = []
        self.features_history = []
        self.max_history = 1000

        # Threading
        self.data_queue = queue.Queue(maxsize=200)

        # Statistics
        self.stats = {
            "total_readings": 0,
            "healthy_count": 0,
            "pest_count": 0,
            "water_count": 0,
            "start_time": None
        }

    # ================================
    # Model Loader
    # ================================
    def load_model(self):
        try:
            base_dir = os.path.dirname(__file__)
            model_path = os.path.join(base_dir, "models", "pvdf_plant_model_latest.pkl")
            scaler_path = os.path.join(base_dir, "models", "scaler_latest.pkl")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                print("✅ AI Model loaded successfully")
                return model, scaler, True
            else:
                print("❌ Model files not found")
                print(model_path)
                print(scaler_path)
                return None, None, False

        except Exception as e:
            print(f"❌ Model load error: {e}")
            return None, None, False

    # ================================
    # ESP32 / Simulation Connection
    # ================================
    def connect(self):
        if self.port is None:
            print("🔄 Simulation mode enabled (no ESP32)")
            return True

        try:
            import serial
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.ser.reset_input_buffer()
            print(f"✅ Connected to ESP32 on {self.port}")
            return True
        except Exception as e:
            print(f"❌ Serial connection failed: {e}")
            return False

    def disconnect(self):
        if self.ser:
            self.ser.close()
            print("🔴 Serial disconnected")

    # ================================
    # Simulated PVDF Data
    # ================================
    def generate_simulated_data(self):
        features = np.random.normal(loc=12, scale=4, size=8).tolist()

        if self.loaded:
            X = np.array(features).reshape(1, -1)
            Xs = self.scaler.transform(X)
            prediction = int(self.model.predict(Xs)[0])
            probs = self.model.predict_proba(Xs)[0]
            confidence = max(probs) * 100
        else:
            prediction = 0
            confidence = 50

        return {
            "timestamp": time.time(),
            "features": features,
            "status_code": prediction,
            "health_score": confidence,
            "confidence": confidence,
            "datetime": datetime.now()
        }

    # ================================
    # Serial Reader Thread
    # ================================
    def read_serial_thread(self):
        while self.running:
            try:
                if self.ser and self.ser.in_waiting:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    parts = line.split(",")
                    if len(parts) >= 8:
                        features = [float(x) for x in parts[:8]]
                        X = np.array(features).reshape(1, -1)
                        Xs = self.scaler.transform(X)
                        prediction = int(self.model.predict(Xs)[0])
                        probs = self.model.predict_proba(Xs)[0]
                        confidence = max(probs) * 100

                        data = {
                            "timestamp": time.time(),
                            "features": features,
                            "status_code": prediction,
                            "health_score": confidence,
                            "confidence": confidence,
                            "datetime": datetime.now()
                        }

                        self.data_queue.put(data)

                time.sleep(0.01)
            except Exception as e:
                print(f"Serial read error: {e}")
                time.sleep(0.1)

    # ================================
    # Display Console Status
    # ================================
    def display_status(self, latest):
        os.system("cls" if os.name == "nt" else "clear")

        status_map = {
            0: "✅ HEALTHY",
            1: "⚠️ PEST STRESS",
            2: "💧 WATER STRESS"
        }

        print("=" * 60)
        print("🌿 REAL-TIME PLANT HEALTH MONITOR")
        print("=" * 60)
        print(f"Status       : {status_map.get(latest['status_code'], 'UNKNOWN')}")
        print(f"Health Score : {latest['health_score']:.1f}%")
        print(f"Confidence   : {latest['confidence']:.1f}%")
        print(f"Time         : {latest['datetime'].strftime('%H:%M:%S')}")
        print("-" * 60)

        for i, v in enumerate(latest["features"]):
            print(f"Feature {i+1}: {v:.2f}")

        elapsed = time.time() - self.stats["start_time"]
        print("-" * 60)
        print(f"Samples   : {self.stats['total_readings']}")
        print(f"Elapsed   : {elapsed:.1f}s")
        print("=" * 60)

    # ================================
    # Main Run Loop
    # ================================
    def run(self, duration=None):

        if not self.connect():
            return

        self.running = True
        self.stats["start_time"] = time.time()

        if self.port is not None:
            t = threading.Thread(target=self.read_serial_thread, daemon=True)
            t.start()

        print("\n🚀 Monitoring started (Ctrl+C to stop)\n")

        try:
            last_display = 0

            while self.running:
                if self.port is None:
                    data = self.generate_simulated_data()
                    self.data_queue.put(data)
                    time.sleep(0.5)

                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    self.data_history.append(data)

                    self.stats["total_readings"] += 1
                    if data["status_code"] == 0:
                        self.stats["healthy_count"] += 1
                    elif data["status_code"] == 1:
                        self.stats["pest_count"] += 1
                    elif data["status_code"] == 2:
                        self.stats["water_count"] += 1

                    if time.time() - last_display > 0.5:
                        self.display_status(data)
                        last_display = time.time()

                if duration and time.time() - self.stats["start_time"] > duration:
                    break

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")

        finally:
            self.running = False
            self.disconnect()
            print("👋 Goodbye")

# ================================
# Main Entry
# ================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None, help="ESP32 COM port (leave empty for simulation)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=int)
    args = parser.parse_args()

    monitor = RealTimePVDFMonitor(port=args.port, baudrate=args.baud)
    monitor.run(duration=args.duration)

if __name__ == "__main__":
    main()
