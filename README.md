# Art Timelapse
This program lets you create efficient timelapses of your drawing.

## Why
I made this program to be used primarily with Paint Tool SAIv2, (which does not have any builtin timelapse tool), however it can be used with any other program.
Previously I was reluctant to record drawing videos with OBS Studio, because I would have to do post-processing on every video:
Cutting dead air from the videos; Concatenating separate recordings together; Speeding up the resulting video to make a timelapse.
While it is possible to mitigate these issues when using OBS Studio, I wanted a more accurate solution that works sort of like Clip Studio's timelapse feature.
By recording the end of each stroke of a drawing, you get the largest amount of information about the timelapse without any redundancy, which can end up creating compcat videos with minimal editing.

## Usage
- When you run the program, it will ask you to click on the window which you want to start capturing.
  - If using Paint Tool SAI, click inside of the drawing area. If you are using Paint Tool SAI with Wine, you can use the `--drag-grab` option to drag a rectangle area to record.
- The program will only record a new video frame after you have finished a click that had started in the target window/area.
  - Because of this, you may leave the program running indefinitely and not have to worry about pausing recording.
- Keystrokes will not create new video frames.
- By default, frames are recorded to a zip file to be processed later (such as if the recorded area changes size). The frame data can take up a lot of space by itself, but the exported video will be fairly small.
- You can choose to record directly to a video file by specifying a `--video-file <filename>`, but it will only record a single frame size. This can save some storage space. You can run with `--convert` and optionally `--export-time-limit <seconds>` to shrink the previously recorded video down to a shorter time.
- You can choose to record from a specific PSD file by specifying `--psd-file <filename>`, in which case a frame will be captured every time the PSD file is finished being written to (such as after saving).
- You can export from a frame data file with `--export` and specifying the frame data file `--frame-data <filename>`. By default the video will try to be made no longer than 60 seconds, like a timelapse, but you can override it with `--export-time-limit <seconds>`. Set it to 0 to let the video time limit be infinite (at 30 FPS).

## Installation from source
- Install python: https://www.python.org/downloads/
- Install dependencies: `pip install -r requirements.txt`
- See arguments with: `python art_timelapse.py`
- If exporting doesn't work, you may also need to install ffmpeg: https://www.gyan.dev/ffmpeg/builds/#release-builds
  - You can add the bin directory to your path, or copy ffmpeg.exe to the script folder.

## Prebuilt executable
- You can find a built exe in the releases page instead of going through the installation, although it may still depend on ffmpeg: https://github.com/cromachina/art-timelapse/releases
- I haven't tested these on their respective platforms. They may have issues loading Tk.

https://github.com/cromachina/art-timelapse/assets/82557197/3e10a9d4-d855-4e91-8070-8f21aa9c350c

