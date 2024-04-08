import numpy as np

from win32api import EnumDisplayMonitors, GetMonitorInfo

def get_monitor_resolution():
    monitors = EnumDisplayMonitors(None, None)
    resolutions = []
    for monitor in monitors:
        monitor_info = GetMonitorInfo(monitor[0])
        left, top, right, bottom = monitor_info['Monitor']
        monitor_resolution = [right - left, bottom - top]
        resolutions.append(monitor_resolution)
    
    max_index = np.argmax([r[0] for r in resolutions])
    return resolutions[max_index]