import argparse
import time
from pathlib import Path
import os
import zipfile
from zipfile import ZipFile

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

def get_max_size(frames_file:ZipFile):
    text = frames_file.comment.decode()
    if text == '':
        return (0, 0)
    else:
        return tuple(map(int, text.split()))

def set_max_size(frames_file:ZipFile, size):
    frames_file.comment = ' '.join(map(str, size)).encode()

def update_max_size(frames_file:ZipFile, old_size, new_size):
    new_max_size = tuple(max(a, b) for a, b in zip(old_size, new_size))
    if any(map(lambda x: x[0] < x[1], zip(old_size, new_max_size))):
        set_max_size(frames_file, new_max_size)
    return new_max_size

def nth_counter(nth):
    counter = 1
    while True:
        if counter == 1:
            counter = nth
            yield True
        else:
            counter -= 1
            yield False

def run_export(frame_data:Path, time_limit):
    with ZipFile(frame_data, 'r') as zfile:
        frames = zfile.namelist()
        frames.sort()
        frame_count = len(frames)
        fps = 30
        nth_frame = 1
        if time_limit is not None:
            target_frames = time_limit * float(fps)
            if frame_count > target_frames:
                nth_frame = np.floor(frame_count / target_frames)
        size = tuple(map(even_dim, get_max_size(zfile)))
        video_file = Path(frame_data).with_suffix('.mp4')
        print(f'Exporting {video_file}')
        video_writer = cv2.VideoWriter(str(video_file), cv2.VideoWriter.fourcc(*'mp4v'), fps, size)
        counter = nth_counter(nth_frame)
        for frame in frames:
            if not next(counter):
                continue
            with zfile.open(frame, 'r') as mfile:
                img = Image.open(mfile)
                if img.size != size:
                    img = ImageOps.pad(img, size)
                frame_data = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                video_writer.write(frame_data)
        video_writer.release()
        print('Finished exporting')

class InputTracker():
    def __init__(self, target_window, callback):
        self.target_window = target_window
        self.callback = callback
        self.click_started_in_window = False

    def click_track(self, x, y, button, is_pressed):
        if is_pressed:
            window = pywinctl.getTopWindowAt(x, y)
            self.click_started_in_window = window.getHandle() == self.target_window.getHandle()
        if not is_pressed and self.click_started_in_window:
            self.click_started_in_window = False
            self.callback()

    def start(self):
        self.mouse_listener = pynput.mouse.Listener(on_click=self.click_track)
        self.mouse_listener.start()

    def stop(self):
        self.mouse_listener.stop()
        self.mouse_listener.join()

def run_capture(frame_data:Path, nth_frame):
    with ZipFile(frame_data, 'a', zipfile.ZIP_DEFLATED) as zfile:
        print('Click on a subwindow to start recording')
        window = get_window_from_click_blocking()
        max_size = get_max_size(zfile)
        counter = nth_counter(nth_frame)
        def capture_frame():
            nonlocal max_size
            if not next(counter):
                return
            img = pyscreenshot.grab(bbox=window.bbox)
            max_size = update_max_size(zfile, max_size, img.size)
            with zfile.open(str(time.time_ns()), 'w') as mfile:
                img.save(mfile, format='jpeg', quality=95)
        input_tracker = InputTracker(window, capture_frame)
        input_tracker.start()
        print(f'Now recording window: {window.title}')
        print(f'Frame data: {frame_data}')
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

def run_psd_capture(frame_data:Path, target_psd:Path, size_limit):
    with ZipFile(frame_data, 'a', zipfile.ZIP_DEFLATED) as zfile:
        max_size = get_max_size(zfile)
        def capture_frame():
            nonlocal max_size
            img = PSDImage.open(target_psd).composite()
            img.thumbnail((size_limit, size_limit))
            max_size = update_max_size(zfile, max_size, img.size)
            with zfile.open(str(time.time_ns()), 'w') as mfile:
                img.save(mfile, format='jpeg', quality=95)
        observer = Observer()
        observer.schedule(FileEventHandler(target_psd, capture_frame), target_psd.parent)
        observer.start()
        print(f'Now recording PSD: {target_psd}')
        print(f'Frame data: {frame_data}')
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
    parser.add_argument('--frame-data', help='Name of the file to store frames in.')
    parser.add_argument('--export', action='store_true', help='Export the given frame data file to an MP4.')
    parser.add_argument('--psd_file', help='Instead of screen recording, record the specified PSD file every time it is written to.')
    parser.add_argument('--size_limit', type=int, default=1000, help='Limit the pixel size from a recorded PSD file.')
    parser.add_argument('--export-time-limit', type=float, default=60, help='Compress the play time of the exported MP4 file in seconds.')
    parser.add_argument('--nth-frame', type=int, default=1, help='For screen recording, record only every Nth frame.')
    args = parser.parse_args()
    no_frame_data = args.frame_data is None
    if no_frame_data:
        args.frame_data = f'{time.time_ns()}.frames'
    args.frame_data = Path(args.frame_data)
    args.frame_data.parent.mkdir(parents=True, exist_ok=True)
    if args.export:
        if no_frame_data:
            print('frame-data must be specified for export')
        else:
            run_export(args.frame_data, args.export_time_limit)
    elif args.psd_file is not None:
        args.psd_file = Path(args.psd_file)
        run_psd_capture(args.frame_data, args.psd_file, args.size_limit)
    else:
        run_capture(args.frame_data, args.nth_frame)

if __name__ == '__main__':
    main()
