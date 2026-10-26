"""Black box for the laptop crashes: writes the temperatures, load, power and memory once per second.

Every line is forced to the disk (fsync), so after a hard crash the last seconds are still in the file.
Run it in a second terminal while you play, and stop it with Ctrl+C:

    python tools/crash_log.py            (writes ~/just_dance_crash_log.csv)

After a crash: look at the last lines of the file (tail -n 30 ~/just_dance_crash_log.csv).
"""
import glob
import os
import sys
import time
from pathlib import Path

LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "just_dance_crash_log.csv"


def read(path, scale=1.0):
    try:
        return round(int(Path(path).read_text().strip()) / scale, 1)
    except (OSError, ValueError):
        return ""


def hwmon(name):
    for folder in glob.glob("/sys/class/hwmon/hwmon*"):
        if Path(folder, "name").read_text().strip() == name:
            return folder
    return None


def cpu_times():
    fields = [int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
    return fields[3] + fields[4], sum(fields)          # idle (+ iowait), total


def game_memory_mb():
    """Memory of the game (python ... main.py, or the JustDance program), in MB."""
    total = 0
    for status in glob.glob("/proc/[0-9]*/status"):
        try:
            cmdline = Path(status).with_name("cmdline").read_bytes().replace(b"\0", b" ")
            if b"main.py" in cmdline or b"JustDance" in cmdline:
                for line in Path(status).read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1]) // 1024
        except OSError:
            pass
    return total


def main():
    cpu, gpu, fans = hwmon("k10temp"), hwmon("amdgpu"), hwmon("lenovo_wmi_other")
    busy = (glob.glob("/sys/class/drm/card*/device/gpu_busy_percent") or [""])[0]
    header = "time,cpu_temp_C,gpu_temp_C,cpu_load_%,gpu_load_%,chip_power_W,fan1_rpm,fan4_rpm,cpu_MHz,mem_free_MB,game_mem_MB\n"
    new = not LOG.exists()
    with open(LOG, "a") as out:
        if new:
            out.write(header)
        out.write(f"# started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        idle0, total0 = cpu_times()
        print(f"writing to {LOG} every second (Ctrl+C stops)")
        while True:
            time.sleep(1.0)
            idle1, total1 = cpu_times()
            load = round(100 * (1 - (idle1 - idle0) / max(1, total1 - total0)), 1)
            idle0, total0 = idle1, total1
            mhz = [float(line.split(":")[1]) for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("cpu MHz")]
            mem_free = next(int(line.split()[1]) // 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                            if line.startswith("MemAvailable"))
            row = [time.strftime("%H:%M:%S"),
                   read(f"{cpu}/temp1_input", 1000) if cpu else "",
                   read(f"{gpu}/temp1_input", 1000) if gpu else "",
                   load, read(busy) if busy else "",
                   (read(f"{gpu}/power1_input", 1e6) or read(f"{gpu}/power1_average", 1e6)) if gpu else "",
                   read(f"{fans}/fan1_input") if fans else "", read(f"{fans}/fan4_input") if fans else "",
                   round(sum(mhz) / len(mhz)) if mhz else "", mem_free, game_memory_mb()]
            out.write(",".join(str(v) for v in row) + "\n")
            out.flush()
            os.fsync(out.fileno())                        # on the disk now: survives a hard crash


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
