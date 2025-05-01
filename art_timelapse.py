import argparse
import time
from pathlib import Path
import os

import cv2
import pynput
import pyscreenshot
import pywinctl
import numpy as np
from PIL import Image, ImageOps
from psd_tools import PSDImage
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

def get_window_from_click_blocking():
    window = None
    def on_click(x, y, button, is_pressed):
        nonlocal window
        if is_pressed:
            window = pywinctl.getTopWindowAt(x, y)
            mouse_listener.stop()
    mouse_listener = pynput.mouse.Listener(on_click=on_click)
    mouse_listener.start()
    mouse_listener.join()
    return window

def even_dim(a):
    return a if a % 2 == 0 else a - 1

metadata = 'metadata'

def get_max_size(frames_folder:Path):
    try:
        with open(frames_folder / metadata, 'r') as f:
            return tuple(map(int, f.read().split()))
    except:
        return (0, 0)

def set_max_size(frames_folder:Path, size):
    with open(frames_folder / metadata, 'w') as f:
        f.write(' '.join(map(str, size)))

def update_max_size(frames_folder:Path, old_size, new_size):
    new_max_size = tuple(max(a, b) for a, b in zip(old_size, new_size))
    if any(map(lambda x: x[0] < x[1], zip(old_size, new_max_size))):
        set_max_size(frames_folder, new_max_size)
    return new_max_size

def run_export(frames_folder:Path, time_limit):
    frames = list(filter(lambda x: x != metadata, os.listdir(frames_folder)))
    frames.sort()
    frame_count = len(frames)
    fps = 30
    nth_frame = 1
    if time_limit is not None:
        target_frames = time_limit * float(fps)
        print(target_frames)
        if frame_count > target_frames:
            nth_frame = np.floor(frame_count / target_frames)
    size = tuple(map(even_dim, get_max_size(frames_folder)))
    video_file = Path(frames_folder).with_suffix('.mp4')
    print(f'Exporting {video_file}')
    video_writer = cv2.VideoWriter(str(video_file), cv2.VideoWriter.fourcc(*'mp4v'), fps, size)
    counter = 1
    for frame in frames:
        if counter == 1:
            counter = nth_frame
        else:
            counter -= 1
            continue
        img = Image.open(frames_folder / frame)
        if img.size != size:
            img = ImageOps.pad(img, size)
        frame_data = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        video_writer.write(frame_data)
    video_writer.release()
    print('Finished exporting')

class InputTracker():
    def __init__(self, target_window, click_callback):
        self.target_window = target_window
        self.click_callback = click_callback
        self.click_started_in_window = False

    def click_track(self, x, y, button, is_pressed):
        if is_pressed:
            window = pywinctl.getTopWindowAt(x, y)
            self.click_started_in_window = window.getHandle() == self.target_window.getHandle()
        if not is_pressed and self.click_started_in_window:
            self.click_started_in_window = False
            self.click_callback()

    def start(self):
        self.mouse_listener = pynput.mouse.Listener(on_click=self.click_track)
        self.mouse_listener.start()

    def stop(self):
        self.mouse_listener.stop()
        self.mouse_listener.join()

def run_capture(frames_folder:Path):
    print('Click on a subwindow to start recording')
    window = get_window_from_click_blocking()
    max_size = get_max_size(frames_folder)
    def capture_frame():
        nonlocal max_size
        img = pyscreenshot.grab(bbox=window.bbox)
        max_size = update_max_size(frames_folder, max_size, img.size)
        img.save(frames_folder / f'{time.time_ns()}', format='jpeg', quality=95)
    input_tracker = InputTracker(window, capture_frame)
    input_tracker.start()
    print(f'Now recording window: {window.title}')
    print(f'Frames folder: {frames_folder}')
    print('Press Ctrl+C here to stop')
    try:
        while True:
            time.sleep(0.1)
            if not window.isAlive:
                print('Window lost; Stopping now')
                break
    except KeyboardInterrupt:
        pass
    print('Stopping tracking')
    input_tracker.stop()
    print('Finished recording')

class FileEventHandler(FileSystemEventHandler):
    def __init__(self, target, callback):
        self.target = target
        self.callback = callback
        super().__init__()

    def on_closed(self, event):
        if Path(event.src_path) == self.target:
            self.callback()

def run_psd_capture(frames_folder:Path, target_psd:Path, size_limit):
    max_size = get_max_size(frames_folder)
    def capture_frame():
        nonlocal max_size
        img = PSDImage.open(target_psd).composite()
        img.thumbnail((size_limit, size_limit))
        max_size = update_max_size(frames_folder, max_size, img.size)
        img.save(frames_folder / f'{time.time_ns()}', format='jpeg', quality=95)
    observer = Observer()
    observer.schedule(FileEventHandler(target_psd, capture_frame), target_psd.parent)
    observer.start()
    print(f'Now recording PSD: {target_psd}')
    print(f'Frames folder: {frames_folder}')
    print('Press Ctrl+C here to stop')
    try:
        while observer.is_alive():
            observer.join(1)
    except KeyboardInterrupt:
        pass
    observer.stop()
    observer.join()
    print('Finished recording')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames-folder')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--psd_file')
    parser.add_argument('--size_limit', type=int, default=1000)
    parser.add_argument('--export-time-limit', type=float, default=60)
    args = parser.parse_args()
    if args.frames_folder is None:
        args.frames_folder = f'{time.time_ns()}.frames'
    args.frames_folder = Path(args.frames_folder)
    args.frames_folder.mkdir(parents=True, exist_ok=True)
    if args.export:
        run_export(args.frames_folder, args.export_time_limit)
    elif args.psd_file is not None:
        args.psd_file = Path(args.psd_file)
        run_psd_capture(args.frames_folder, args.psd_file, args.size_limit)
    else:
        run_capture(args.frames_folder)

if __name__ == '__main__':
    main()
