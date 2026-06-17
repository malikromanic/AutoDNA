# -*- coding: utf-8 -*-
"""
BeamNG data collection script.
Drives AI on a map, records IMU sensor data and generates labels
in the same format as real-world recordings.
 
Requirements:
    pip install beamngpy numpy
 
Usage:
    1. Start BeamNG.tech (or let this script launch it)
    2. Run this script
    3. Outputs: sim_LOGXXX.npz + sim_LOGXXX_labels.json
       Copy these to parsed_data/ and labeled_data_json/ to use with existing pipeline.
"""
 
import json
import time
import math
import numpy as np
from pathlib import Path
from datetime import datetime
 
try:
    from beamngpy import BeamNGpy, Scenario, Vehicle
    from beamngpy.sensors import AdvancedIMU, State, Electrics
except ImportError:
    raise ImportError("Install beamngpy: pip install beamngpy")
 
 
# ── configuration ─────────────────────────────────────────────────────────────
 
BEAMNG_HOME   = r'C:\BeamNG\BeamNG.tech.v0.37.6.0'        # adjust to your install
#BEAMNG_USER   = r'C:\Users\mihal\Documents\BeamNG.tech' # adjust to your user dir
OUTPUT_DIR    = Path('sim_data')
LOG_PREFIX    = 'sim_LOG'
 
# west_coast_usa has lots of hills and winding roads — good for data collection
# alternatives with hills: 'italy', 'small_island'
MAP           = 'west_coast_usa'
VEHICLE_MODEL = 'etk800'
 
# spawn position with elevation changes nearby
SPAWN_POS     = (-717.0, 101.0, 118.0)
SPAWN_ROT     = (0, 0, 0.3826834, 0.9238795)
 
COLLECTION_DURATION_SEC = 30   # 5 minutes per run — run multiple times for more data
POLL_HZ       = 50              # 50 Hz — matches typical IMU rate
 
# label thresholds — tune these to match your real-world label conventions
TURN_YAW_RATE_THRESHOLD  = 10.0    # deg/s minimum to count as a turn
HILL_PITCH_THRESHOLD     = 1.5    # degrees minimum to count as a hill
TURN_INTENSE_THRESHOLD       = 30.0   # deg/s — oster vs blag
HILL_INTENSE_THRESHOLD   = 8.0    # degrees — oster vs blag
 
MAX_ANGLE_TURN = 100.0
MAX_ANGLE_HILL = 15.0

RAD_TO_DEG = 57.2957795
 
# ── helpers ───────────────────────────────────────────────────────────────────


def quat_to_pitch(rotation):
    """Extract pitch and yaw from quaternion [x, y, z, w]."""
    x, y, z, w = rotation
    # pitch (rotation around X axis)
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.degrees(math.asin(sinp))
    return pitch


def find_next_log_index(output_dir):
    existing = list(output_dir.glob(f'{LOG_PREFIX}*.npz'))
    if not existing:
        return 1
    indices = []
    for f in existing:
        try:
            indices.append(int(f.stem.replace(LOG_PREFIX, '')))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 1
 
 
def frames_to_npz(frames, output_path):
    """
    Save IMU frames as .npz compatible with load_sensor_data() in preprocessing.py
    Each sensor array has shape (N, 4): [timestamp_ms, x, y, z]
    Mag is zeros — not available in BeamNG but included to avoid pipeline errors.
    """
    n   = len(frames)
    ts  = np.array([f['t'] * 1000.0 for f in frames], dtype=np.float32)  # ms
 
    accel = np.column_stack([
        ts,
        np.array([f['accel_x'] for f in frames], dtype=np.float32),
        np.array([f['accel_y'] for f in frames], dtype=np.float32),
        np.array([f['accel_z'] for f in frames], dtype=np.float32),
    ])
    gyro = np.column_stack([
        ts,
        np.array([f['gyro_x'] for f in frames], dtype=np.float32),
        np.array([f['gyro_y'] for f in frames], dtype=np.float32),
        np.array([f['gyro_z'] for f in frames], dtype=np.float32),
    ])
    # mag zeros — shape matches accel/gyro so pipeline doesn't break
    mag = np.column_stack([
        ts,
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    ])
 
    np.savez(str(output_path), accel=accel, gyro=gyro, mag=mag)
    print(f"Saved sensor data: {output_path.name}  ({n} samples @ {POLL_HZ}Hz)")
 
 
def classify_frame(yaw_rate, pitch):
    """Classify one frame into turn/hill/straight."""
    turn = None
    if abs(yaw_rate) > TURN_YAW_RATE_THRESHOLD:
        turn = {
            'dir':      'right' if yaw_rate > 0 else 'left',
            'angleDeg': round(float(min(abs(yaw_rate), MAX_ANGLE_TURN)), 1),
            'intensity': 'oster' if abs(yaw_rate) > TURN_INTENSE_THRESHOLD else 'blag',
        }
 
    hill = None
    if abs(pitch) > HILL_PITCH_THRESHOLD:
        hill = {
            'dir':      'up' if pitch > 0 else 'down',
            'angleDeg': round(float(min(abs(pitch), MAX_ANGLE_HILL)), 1),
            'intensity': 'oster' if abs(pitch) > HILL_INTENSE_THRESHOLD else 'blag',
        }
 
    straight = (turn is None) and (hill is None)
    return turn, hill, straight
 
 
def frames_to_label_segments(frames):
    """
    Convert per-frame states into contiguous segments — same format as
    manual JSON labels with t_start, t_end, turn, hill, straight fields.
    """
    if not frames:
        return []
 
    segments = []
    seg_t0   = frames[0]['t']
    seg_t1   = frames[0]['t']
    seg_turn, seg_hill, seg_straight = classify_frame(
        frames[0]['yaw_rate'], frames[0]['pitch']
    )
 
    for f in frames[1:]:
        turn, hill, straight = classify_frame(f['yaw_rate'], f['pitch'])
 
        # same state = same turn direction and same hill direction
        same = (
            (turn['dir']  if turn  else None) == (seg_turn['dir']  if seg_turn  else None) and
            (hill['dir']  if hill  else None) == (seg_hill['dir']  if seg_hill  else None)
        )
 
        if same:
            seg_t1 = f['t']
            # keep angle updated to latest measurement
            if turn and seg_turn:
                seg_turn['angleDeg'] = round(float(min(abs(f['yaw_rate']), MAX_ANGLE_TURN)), 1)
            if hill and seg_hill:
                seg_hill['angleDeg'] = round(float(min(abs(f['pitch']), MAX_ANGLE_HILL)), 1)
        else:
            segments.append({
                't_start':  float(seg_t0),
                't_end':    float(seg_t1),
                'turn':     seg_turn,
                'hill':     seg_hill,
                'straight': bool(seg_straight),
            })
            seg_t0       = f['t']
            seg_t1       = f['t']
            seg_turn     = turn
            seg_hill     = hill
            seg_straight = straight
 
    # flush last segment
    segments.append({
        't_start':  float(seg_t0),
        't_end':    float(seg_t1),
        'turn':     seg_turn,
        'hill':     seg_hill,
        'straight': bool(seg_straight),
    })
 
    return segments
 
 
def save_labels_json(segments, npz_filename, output_path):
    """Save labels in the autodna-v2-multilayer JSON schema."""
    data = {
        'file':     npz_filename,
        'exported': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        'schema':   'autodna-v2-multilayer',
        'labels':   segments,
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved labels:      {output_path.name}  ({len(segments)} segments)")
 
 
# ── main collection ───────────────────────────────────────────────────────────
 
def collect(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    log_idx   = find_next_log_index(output_dir)
    log_name  = f'{LOG_PREFIX}{log_idx:03d}'
    npz_path  = output_dir / f'{log_name}.npz'
    json_path = output_dir / f'{log_name}_labels.json'
 
    print(f"Connecting to BeamNG.tech...")
    bng = BeamNGpy('localhost', 64256, home=BEAMNG_HOME)    #, user=BEAMNG_USER
    #bng = BeamNGpy("localhost", 25252)
    bng.open(launch=True)
 
    try:
        scenario = Scenario(MAP, 'autodna_collection')
        # setup — before scenario.make(bng):
        vehicle = Vehicle('ego', model=VEHICLE_MODEL, license='AUTODNA')
        #vehicle.attach_sensor('state', State())
        #imu = AdvancedIMU(...)  # or IMU depending on what's available
        
        scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
        scenario.make(bng)
        bng.load_scenario(scenario)
        bng.start_scenario()
 
        electrics = Electrics()
        state = State()
        vehicle.sensors.attach('electrics', electrics)
        vehicle.sensors.attach('state_sensor', state)
        imu = AdvancedIMU('imu', bng, vehicle)
        
        # enable AI to drive the whole map
        vehicle.ai.set_mode('span')
        vehicle.ai.set_speed(13.0, mode='set')   # ~47 km/h
 
        print(f"AI driving. Collecting {COLLECTION_DURATION_SEC}s of data...")
        print(f"Saving to: {npz_path.name}")
 
        frames  = []
        t_start = time.time()
        
        c = 0
        while True:
            elapsed = time.time() - t_start
            if elapsed >= COLLECTION_DURATION_SEC:
                print(f"\nDone ({elapsed:.1f}s, {len(frames)} frames)")
                break
 
            bng.step(1, wait=True)
            
            vehicle.sensors.poll()
            e = electrics.data
            s = state.data
            imu_data = imu.poll()
            if c == 0:
                print(imu_data)
                print(imu_data.keys())
                c = 1
                
            if e and s:
                # AdvancedIMU keys — print imu_data.keys() on first frame if unsure
                frames.append({
                    't':        elapsed,
                    # accSmooth with gravity=True matches real accelerometer (gravity included)
                    'accel_x':  float(imu_data[0.0]['accSmooth'][0]),
                    'accel_y':  float(imu_data[0.0]['accSmooth'][1]),
                    'accel_z':  float(imu_data[0.0]['accSmooth'][2]),
                    # angVel converted to deg/s matches real gyro output
                    'gyro_x':   float(imu_data[0.0]['angVel'][0]) * RAD_TO_DEG,
                    'gyro_y':   float(imu_data[0.0]['angVel'][1]) * RAD_TO_DEG,
                    'gyro_z':   float(imu_data[0.0]['angVel'][2]) * RAD_TO_DEG,
                    # labels from quaternion for accurate angle thresholds
                    'yaw_rate': float(imu_data[0.0]['angVel'][2]) * RAD_TO_DEG,
                    'pitch':    quat_to_pitch(s['rotation']), 
                })
 
            if len(frames) == 1:
                print(f"Electrics keys: {list(e.keys())}")
                print(f"State keys:     {list(s.keys())}")
 
            print(f"\r  t={elapsed:6.1f}s  frames={len(frames):5d}", end='', flush=True)
 
        # save sensor data as .npz
        frames_to_npz(frames, npz_path)
 
        # generate and save labels as .json
        segments = frames_to_label_segments(frames)
        save_labels_json(segments, npz_path.name, json_path)
 
        # summary
        n_turns    = sum(1 for s in segments if s['turn'])
        n_hills    = sum(1 for s in segments if s['hill'])
        n_straight = sum(1 for s in segments if s['straight'])
        print(f"\nSummary:")
        print(f"  turns:    {n_turns}")
        print(f"  hills:    {n_hills}")
        print(f"  straight: {n_straight}")
        print(f"\nTo use with existing pipeline, copy files to:")
        print(f"  {npz_path.name}       → parsed_data/")
        print(f"  {json_path.name}  → labeled_data_json/")
 
    finally:
        bng.close()
        print("BeamNG closed.")
 
 
if __name__ == '__main__':
    collect()