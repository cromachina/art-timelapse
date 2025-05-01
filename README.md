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
  - If using Paint Tool SAI, click inside of the drawing area. If you are using Paint Tool SAI with Wine, it may only be able to capture the entire window.
- The program will only record a new video frame after you have finished a click that had started in the target window.
  - Because of this, you may leave the program running indefinitely and not have to worry about pausing recording.
- Keystrokes will not create new video frames.
- You can instead choose to record from a specific PSD file, in which case a frame will be captured every time the PSD file is closed (finished being written to).

## Installation
- Install python: https://www.python.org/downloads/
- Download or clone this repository.
- Install poetry: `pip install poetry`
- Run `poetry install` to install the program and its dependencies.
- Run `poetry run art-timelapse --help` to see command line arguments.
- If recording doesn't work, you may also need to install ffmpeg: https://www.gyan.dev/ffmpeg/builds/#release-builds
  - You can add the bin directory to your path, or copy ffmpeg.exe to the script folder.

## Prebuilt executable
- You can find a built exe in the releases page instead of going through the installation, although it may still depend on ffmpeg: https://github.com/cromachina/art-timelapse/releases

https://github.com/cromachina/art-timelapse/assets/82557197/3e10a9d4-d855-4e91-8070-8f21aa9c350c

