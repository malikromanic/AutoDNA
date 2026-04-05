# -*- coding: utf-8 -*-
"""
BeamNG Sensor Tracker
=====================
 
Collects vehicle telemetry data (speed, fuel consumption, acceleration)
from a running BeamNG.tech simulation using the BeamNGpy API.
 
The script connects to an already-running BeamNG.tech instance, attaches
sensors to the player vehicle, runs a two-phase speed test, and plots
the collected data.
 
 data collected on map West coast USA - Highway
 
:author: mihal
:date: 2026-03-29
"""
 
import numpy as np
import matplotlib.pyplot as plt
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, State
 
 
# ---------------------------------------------------------------------------
# Global telemetry storage
# ---------------------------------------------------------------------------
 
timestamps: list[float] = []
speeds: list[float] = []
fuel_volumes: list[float] = []
fuel_consumption: list[float] = []
accel_x: list[float] = []
accel_y: list[float] = []
accel_z: list[float] = []
 
_prev_time: float = 0.0
_prev_fuel_volume: float | None = None
 
 
# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
 
def drive_at_speed(
    bng: BeamNGpy,
    electrics: Electrics,
    state: State,
    vehicle: Vehicle,
    target_speed: float,
    duration: float,
) -> None:
    """Drive the vehicle at a constant target speed for a given duration.
 
    Uses the built-in BeamNG AI in ``limit`` mode to maintain the requested
    speed.  Telemetry is sampled every physics step and appended to the
    module-level storage lists.
 
    :param bng: Connected :class:`BeamNGpy` instance.
    :param electrics: Attached :class:`Electrics` sensor for the vehicle.
    :param state: Attached :class:`State` sensor providing simulation time.
    :param vehicle: The :class:`Vehicle` being controlled.
    :param target_speed: Desired speed in metres per second.
    :param duration: How long (in simulation seconds) to hold this speed.
    """
    global _prev_fuel_volume
 
    print(f"Setting target speed to: {target_speed:.1f} m/s "
          f"({target_speed * 3.6:.1f} km/h)")
 
    vehicle.ai.set_mode("span")
    vehicle.ai.set_speed(target_speed, mode="limit")
    vehicle.ai.set_aggression(1)
 
    vehicle.sensors.poll()
    current_sim_time: float = state.data["time"]
    end_sim_time: float = current_sim_time + duration
    last_sim_time: float = current_sim_time
 
    while current_sim_time < end_sim_time:
        vehicle.sensors.poll()
        e = electrics.data
 
        current_speed: float = e.get("wheelspeed", 0.0)
        fuel_vol: float = e.get("fuel_volume", 0.0)
        current_sim_time = state.data["time"]
 
        dt: float = current_sim_time - last_sim_time
        if dt > 0 and _prev_fuel_volume is not None and fuel_vol is not None:
            consumption: float = (_prev_fuel_volume - fuel_vol) / dt
        else:
            print("fail", dt, _prev_fuel_volume, fuel_vol)
            consumption = 0.0
 
        timestamps.append(current_sim_time)
        speeds.append(current_speed)
        fuel_volumes.append(fuel_vol)
        fuel_consumption.append(consumption)
        accel_x.append(e.get("accXSmooth", 0.0))
        accel_y.append(e.get("accYSmooth", 0.0))
        accel_z.append(e.get("accZSmooth", 0.0))
 
        last_sim_time = current_sim_time
        _prev_fuel_volume = fuel_vol
 
        bng.control.step(1)
 
 
def run_test_scenario(
    bng: BeamNGpy,
    electrics: Electrics,
    state: State,
    vehicle: Vehicle,
    speed_1: float,
    speed_2: float,
    total_test_time: float,
) -> None:
    """Run a two-phase speed test followed by a brief stop phase.
 
    The total test time is split evenly between ``speed_1`` and ``speed_2``.
    After both phases, the vehicle is brought to a stop for 5 simulation
    seconds.
 
    :param bng: Connected :class:`BeamNGpy` instance.
    :param electrics: Attached :class:`Electrics` sensor for the vehicle.
    :param state: Attached :class:`State` sensor providing simulation time.
    :param vehicle: The :class:`Vehicle` being controlled.
    :param speed_1: Target speed for the first phase, in metres per second.
    :param speed_2: Target speed for the second phase, in metres per second.
    :param total_test_time: Total duration of both phases combined, in
        simulation seconds.
    """
    print("Test running...")
    slice_duration: float = total_test_time / 2.0
 
    drive_at_speed(bng, electrics, state, vehicle, speed_1, slice_duration)
    drive_at_speed(bng, electrics, state, vehicle, speed_2, slice_duration)
    drive_at_speed(bng, electrics, state, vehicle, 0.0, 5.0)
 
    print("Test finished.")
 
 
def build_data_dict() -> dict[str, np.ndarray]:
    """Convert module-level telemetry lists into a dictionary of NumPy arrays.
 
    :returns: Dictionary with the following keys:
 
        * ``time`` – simulation timestamps (s)
        * ``speed_ms`` – vehicle speed (m/s)
        * ``speed_kmh`` – vehicle speed (km/h)
        * ``fuel_volume`` – remaining fuel (L)
        * ``fuel_consumption`` – instantaneous fuel consumption (L/s)
        * ``accel_x`` – longitudinal acceleration (m/s²)
        * ``accel_y`` – lateral acceleration (m/s²)
        * ``accel_z`` – vertical acceleration (m/s²)
 
    :rtype: dict[str, numpy.ndarray]
    """
    speeds_array = np.array(speeds)
    return {
        "time": np.array(timestamps),
        "speed_ms": speeds_array,
        "speed_kmh": speeds_array * 3.6,
        "fuel_volume": np.array(fuel_volumes),
        "fuel_consumption": np.array(fuel_consumption),
        "accel_x": np.array(accel_x),
        "accel_y": np.array(accel_y),
        "accel_z": np.array(accel_z),
    }
 
 
def plot_telemetry(data: dict[str, np.ndarray]) -> None:
    """Plot the collected telemetry data across four subplots.
 
    All subplots share the same x-axis (simulation time) so that zooming
    one panel synchronises the rest.
 
    :param data: Dictionary of NumPy arrays as returned by
        :func:`build_data_dict`.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
 
    axes[0].plot(data["time"], data["speed_kmh"], color="blue", label="Speed")
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].legend()
    axes[0].grid(True)
 
    axes[1].plot(data["time"], data["fuel_volume"], color="green", label="Fuel volume")
    axes[1].set_ylabel("Fuel (L)")
    axes[1].legend()
    axes[1].grid(True)
 
    axes[2].plot(
        data["time"], data["fuel_consumption"], color="orange", label="Fuel consumption"
    )
    axes[2].set_ylabel("Consumption (L/s)")
    axes[2].legend()
    axes[2].grid(True)
 
    axes[3].plot(data["time"], data["accel_x"], label="Accel X (longitudinal)")
    axes[3].plot(data["time"], data["accel_y"], label="Accel Y (lateral)")
    axes[3].plot(data["time"], data["accel_z"], label="Accel Z (vertical)")
    axes[3].set_ylabel("Acceleration (m/s²)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend()
    axes[3].grid(True)
 
    plt.tight_layout()
    plt.show()
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
def main() -> None:
    """Connect to BeamNG.tech, run the test scenario, and plot the results.
 
    Connects to a BeamNG.tech instance already listening on
    ``localhost:25252``, retrieves the currently active player vehicle,
    attaches the required sensors, and runs :func:`run_test_scenario`.
    After the test the collected data is saved to ``drive_data.npy`` and
    displayed as a four-panel plot.
    """
    bng = BeamNGpy("localhost", 25252)
    bng.open(launch=False)
 
    active_vehicles = bng.vehicles.get_current()
    vehicle = next(iter(active_vehicles.values()))
    vehicle.connect(bng)
 
    electrics = Electrics()
    state = State()
    vehicle.sensors.attach("electrics", electrics)
    vehicle.sensors.attach("state_att", state)
 
    total_test_time: float = 80.0
    speed_phase_1: float = 40.0 / 3.6   # 40 km/h → m/s
    speed_phase_2: float = 10.0 / 3.6   # 10 km/h → m/s
 
    run_test_scenario(
        bng,
        electrics,
        state,
        vehicle,
        speed_phase_1,
        speed_phase_2,
        total_test_time,
    )
 
    data = build_data_dict()
    np.save("drive_data.npy", data)
    print(f"Saved {len(timestamps)} samples to drive_data.npy")
 
    plot_telemetry(data)
    bng.disconnect()
 
 
if __name__ == "__main__":
    main()