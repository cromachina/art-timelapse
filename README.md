# Art Timelapse
This program lets you create efficient timelapses of your drawing.

### Why
I made this program to be used primarily with Paint Tool SAIv2, (which does not have any builtin timelapse tool), however it can be used with any other program.
Previously I was reluctant to record drawing videos with OBS Studio, because I would have to do post-processing on every video:
Cutting dead air from the videos; Concatenating separate recordings together; Speeding up the resulting video to make a timelapse.
While it is possible to mitigate these issues when using OBS Studio, I wanted a more accurate solution that works sort of like Clip Studio's timelapse feature.
By recording the end of each stroke of a drawing, you get the largest amount of information about the timelapse without any redundancy, which can end up creating compact videos with minimal editing.

### Recording
- The program will only record a new video frame after you have finished a click that had started in the target window/area.
  - Because of this, you may leave the program running indefinitely and not have to worry about pausing recording.
- Keystrokes will not create new video frames.
#### SAI memory capture
- If using `--sai`, the program will read frames directly from the running SAI instance.
- It will ask you which opened canvas you want to record.
- It will try to automatically find the window of SAI to track clicks, but if it cannot find the window, you will be asked to click on SAI's window instead.
- It will only record a new frame if it is different from the last frame, as to prevent clicks on other parts of the program from creating redundant frames.
#### PSD capture
- If using `--psd-file <filename>`, a frame will be captured every time the PSD file is finished being written to (such as after saving).
#### Screen capture
- The default mode is to capture an area of the screen.
- When you run the program, it will ask you to click on the window which you want to start capturing.
  - If using Paint Tool SAI with Windows, click inside of the drawing area to automatically capture that subwindow.
  - If you are using Paint Tool SAI with Wine, you can use the `--drag-grab` option to drag a rectangle area to record.
#### Exporting video
- By default, frames are recorded to a zip file to be processed later. The frame data can take up a lot of space by itself, but the exported video will be fairly small.
- You can instead record to videos by specifying `--video`. If the resulting image size to be stored changes, it will automatically cut a new video. Overall, this can save a lot more storage space than the zip file, but it needs FFPMEG to work.
- You can export the saved frame data with `--export` and specifying the frame data file `--frames <filename>`. By default the video will try to be made no longer than 60 seconds, like a typical timelapse, but you can override it with `--export-time-limit <seconds>`. Set it to 0 to have no limit on the export length. If the frame data is a directory, such as when previously using `--video`, it will instead concatenate the videos in the directory together, compressed to the appropriate length.

### Examples
Record from SAI's memory directly and put the outputs as MP4s into a directory called `output`:
```
art-timelapse --sai --frames output --video
```
Interactive setup:
```
Press Ctrl+C here to stop at any time
Opening frames: output
Select a canvas to record (Ctrl+C to cancel):
[1] 20250821.sai2 (F:\home\cro\aux\art\20250821.sai2)
[2] NewCanvas3 ()
Enter index [1-2]:2
Could not find window automatically
Click on a subwindow to start recording. Right click to cancel.
Tracking input for window: Default - Wine desktop
```
Finishing recording (pressed Ctrl+C):
```
^CClosing frames: output
```
Exporting resulting output, you do not have to specify `--video`. The export type is inferred from the `--frames` path:
```
art-timelapse --sai --frames output --export
Press Ctrl+C here to stop at any time
Opening frames: output
100%|████████████████████████████████| 23/23 [00:00<00:00, 116.19frames/s]
```

### Installation from source
- Install python: https://www.python.org/downloads/
- Install project: `pip install -e .`
- See arguments with: `art-timelapse --help`
- If exporting doesn't work, you may also need to install ffmpeg: https://www.gyan.dev/ffmpeg/builds/#release-builds
  - You can add the bin directory to your path, or copy ffmpeg.exe to the script folder.

### Building/installing with Nix
- This project is a Nix flake, so you can run flake commands to interact with the package `nix run`, `nix build`, etc.

### Prebuilt executable
- You can find a built exe in the releases page instead of going through the installation, although it may still depend on ffmpeg: https://github.com/cromachina/art-timelapse/releases
- I haven't tested these on their respective platforms. They may have issues loading Tk.

https://github.com/cromachina/art-timelapse/assets/82557197/3e10a9d4-d855-4e91-8070-8f21aa9c350c

