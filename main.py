import ctypes
import datetime
import time
from ctypes import wintypes

import cv2
import numpy as np
import pynput
import win32con
import win32gui
import win32ui

gdi32 = ctypes.windll.gdi32

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD)
    ]

def get_hwnd_from_click_blocking():
    hwnd = None
    mouse_listener = None

    def on_click(x, y, button, is_pressed):
        nonlocal hwnd
        if is_pressed:
            hwnd = win32gui.WindowFromPoint((x, y))
            mouse_listener.stop()

    mouse_listener = pynput.mouse.Listener(on_click=on_click)
    mouse_listener.start()
    mouse_listener.join()
    return hwnd

def even_dim(a, b):
    c = abs(a - b)
    return c if c % 2 == 0 else c - 1

class ScreenRecorder():
    def __init__(self):
        pass

    def start(self, hwnd, output_file=None):
        self.hwnd = hwnd
        self.root_hwnd = win32gui.GetAncestor(self.hwnd, win32con.GA_ROOT)
        self.window = win32ui.CreateWindowFromHandle(self.hwnd)
        self.dc = self.window.GetDC()
        self.dc_mem = self.dc.CreateCompatibleDC(self.dc)
        rect = self.window.GetClientRect()
        self.width = even_dim(rect[2], rect[0])
        self.height = even_dim(rect[3], rect[1])
        self.bitmap = win32ui.CreateBitmap()
        self.bitmap.CreateCompatibleBitmap(self.dc, self.width, self.height)
        self.dc_mem.SelectObject(self.bitmap)
        info = self.bitmap.GetInfo()
        self.bminfo = BITMAPINFOHEADER()
        self.bminfo.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        self.bminfo.biWidth = self.width
        self.bminfo.biHeight = -self.height
        self.bminfo.biPlanes = info['bmPlanes']
        self.bminfo.biBitCount = info['bmBitsPixel']
        self.buffer = np.empty((self.height, self.width, 4), dtype=np.uint8)

        self.output_file = output_file
        if self.output_file is None:
            date_now = datetime.datetime.now().strftime('%y%m%d%H%M%S')
            self.output_file = f'{date_now}.mp4'
        self.video_writer = cv2.VideoWriter(self.output_file, cv2.VideoWriter.fourcc(*'iYUV'), 30, (self.width, self.height))

    def write_frame(self):
        self.dc_mem.BitBlt((0, 0), (self.width, self.height), self.dc, (0, 0), win32con.SRCCOPY)
        gdi32.GetDIBits(self.dc_mem.GetSafeHdc(), self.bitmap.GetHandle(), 0, self.height, self.buffer.ctypes.data_as(ctypes.c_void_p), self.bminfo, 0)
        self.video_writer.write(self.buffer)

    def stop(self):
        self.video_writer.release()
        try:
            self.dc_mem.DeleteDC()
        except:
            pass
        try:
            self.dc.DeleteDC()
        except:
            pass

class InputTracker():
    def __init__(self, screen_recorder):
        self.screen_recorder = screen_recorder
        self.click_started_in_window = False

    def click_track(self, x, y, button, is_pressed):
        if is_pressed:
            self.click_started_in_window = win32gui.WindowFromPoint((x, y)) == self.screen_recorder.hwnd
        if not is_pressed and self.click_started_in_window:
            self.click_started_in_window = False
            self.screen_recorder.write_frame()

    def start(self):
        self.mouse_listener = pynput.mouse.Listener(on_click=self.click_track)
        self.mouse_listener.start()

    def stop(self):
        self.mouse_listener.stop()
        self.mouse_listener.join()

def main():
    print('Click on a window to start recording')
    hwnd = get_hwnd_from_click_blocking()
    screen_recorder = ScreenRecorder()
    input_tracker = InputTracker(screen_recorder)
    screen_recorder.start(hwnd)
    input_tracker.start()
    print(f'Now recording window: {win32gui.GetWindowText(hwnd)}')
    print(f'Output file: {screen_recorder.output_file}')
    print('Press Ctrl+C here to stop')
    try:
        while True:
            time.sleep(0.1)
            if not win32gui.IsWindow(hwnd):
                print('Window lost; Stopping now')
                break
    except KeyboardInterrupt:
        pass
    input_tracker.stop()
    screen_recorder.stop()
    print('Finished recording')

if __name__ == '__main__':
    main()
