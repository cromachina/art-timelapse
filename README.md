# Art Timelapse
This program lets you create efficient timelapses of your drawing.

## Why
I made this program to be used primarily with Paint Tool SAIv2, (which does not have any builtin timelapse tool), however it can be used with any other program on Windows.
Previously I was reluctant to record drawing videos with OBS Studio, because I would have to do post-processing on every video:
Cutting dead air from the videos; Concatenating separate recordings together; Speeding up the resulting video to make a timelapse.
While it is possible to mitigate these issues when using OBS Studio, I wanted a more accurate solution that works sort of like Clip Studio's timelapse feature.
By recording the end of each stroke of a drawing, you get the largest amount of information about the timelapse without any redundancy, which ends up creating fairly small sized videos.

## Usage
- Make sure the window that you want to record is the resized to the size you want to record at before you start recording.
  - If you resize the window after starting recording, the resulting video will only be the size at the start of recording, and it may not capture everything you want.
- When you run the program, it will ask you to click on the window which you want to start capturing.
  - If using Paint Tool SAI, click inside of the drawing area.
- If you see an error about an unsupported codec, you can ignore it.
- The program will only record a new video frame after you have finished a click that had started in the target window.
  - Because of this, you may leave the program running indefinitely and not have to worry about pausing recording.
- Keystrokes will not create new video frames.

## Installation
- Install python: https://www.python.org/downloads/
- Run `pip install -r requirements.txt` to install dependencies
- Run `python main.py` (or double click on `main.py`) to start the program
- If recording doesn't work, you may also need to install ffmpeg: https://www.gyan.dev/ffmpeg/builds/#release-builds
  - You can add the bin directory to your path, or copy ffmpeg.exe to the script folder.

https://github.com/cromachina/art-timelapse/assets/82557197/3e10a9d4-d855-4e91-8070-8f21aa9c350c

