# Art Timelapse
This program lets you create efficient timelapses of your drawing.

### Why
I made this program to be used primarily with Paint Tool SAIv2, (which does not have any builtin timelapse tool), however it can be used with any other program.
Previously I was reluctant to record drawing videos with OBS Studio, because I would have to do post-processing on every video:
Cutting dead air from the videos; Concatenating separate recordings together; Speeding up the resulting video to make a timelapse.
While it is possible to mitigate these issues when using OBS Studio, I wanted a more accurate solution that works sort of like Clip Studio's timelapse feature.
By recording the end of each stroke of a drawing, you get the largest amount of information about the timelapse without any redundancy, which can end up creating compact videos with minimal editing.

### GUI usage
- The default mode is to run a GUI and most options have tooltips.

### Command line usage
- Use `--cli` to run the program as a command line tool. Use `--help` to see all options.
- Specify `--frames <folder>` recorded data to a directory. If not specified, it will use a timestamp as the file name.
- The default video storage format is `webm` with `vp80` codec to have better color preservation (compared to `mp4`).
  - You may see a warning that `vp80` is not supported for `webm`. This appears to be a bug in OpenCV that you can ignore.
- You can specify a custom `--container <suffix>` and fourcc `--codec <fourcc>` when recording or exporting, whatever is supported by OpenCV and FFMPEG, for example `--container mp4 --codec mp4v`.
  - Incompatible containers and codecs will display errors and might not produce any resulting video file.
- If you specify `--web`, the `container` and `codec` are set to `mp4` and `avc1` respectively. This format is accepted by many websites for upload, but it can reduce color quality. This is best used when exporting (with `--export`). If you really need to save disk space, then recording with `--web` will be very efficient.
- You can export the saved frame data with `--export`. By default the video will try to be made no longer than 60 seconds, like a typical timelapse, but you can override it with `--export-time-limit <seconds>`. Set it to 0 to have no limit on the export length.

| Format | Quality | Size (relative to MP4) | Estimated size (~25k 1000px frames recorded) |
|--|--|--|--|
| mp4 (avc1) | Some colors will look off from the original | 1x | 15 MB |
| webm (vp80) | Colors look more accurate | 10x | 150 MB |

#### SAI memory capture
- If using `--sai`, the program will read frames directly from the running SAI instance.
- It will ask you which opened canvas you want to record.
- It will try to automatically find the window of SAI to track clicks, but if it cannot find the window, you will be asked to click on SAI's window instead.
- It will only record a new frame if it is different from the last frame, as to prevent clicks on other parts of the program from creating redundant frames.
- This mode will not work natively on MacOS because PyMemoryEditor does not have a MacOS implementation. You might be able to run this program in Wine on MacOS to get around this.
- I think that this mode provides the best looking timelapse for SAI.
- This mode has to be updated for each new version of SAI, so it may not be compatible with the latest version of SAI as it's released.
#### PSD capture
- If using `--psd-file <filename>`, a frame will be captured every time the PSD file is finished being written to (such as after saving).
- This mode will make a choppy looking timelapse, depending on how frequently you save your work, but the effect isn't terrible.
#### Screen capture
- This is the default behavior of the program when neither of the above options is specified, which is to capture an area of the screen.
- When you run the program, it will ask you to click on the window which you want to start capturing.
  - If using Paint Tool SAI with Windows, click inside of the drawing area to automatically capture that subwindow.
  - If you are using Paint Tool SAI with Wine, you can use the `--drag-grab` option to drag a rectangular area to record.
- This mode will make a timelapse that looks like a screen recording with software like OBS. Depending on how short you compress the timelapse to, this can look somewhat disorienting, particularly if you zoom or rotate a lot.

### Examples
Record from SAI's memory directly and store the outputs into a directory called `test`:
```
art-timelapse --cli --frames test --sai
```
Interactive setup:
```
Press Ctrl+C here to stop at any time
Writing to frames folder: test; container webm; codec vp80
Select a canvas to record (Ctrl+C to cancel):
[1] NewCanvas16 ()
[2] NewCanvas17 ()
Enter index [1-2]:2
Could not find window automatically
Click on a subwindow to start recording. Right click to cancel.
Tracking input for window: Default - Wine desktop
OpenCV: FFMPEG: tag 0x30387076/'vp80' is not supported with codec id 139 and format 'webm / WebM'
^C
```
Finishing recording (pressed Ctrl+C).

The export type is inferred from the `--frames` path:
```
art-timelapse --cli --frames test --sai --export --web
```
Output:
```
Press Ctrl+C here to stop at any time
Reading from frames folder: test
Writing to video: test.mp4 (avc1)
100%|████████████████████████████| 26/26 [00:00<00:00, 184.61frames/s]
```

### Windows prebuilt executable (pyinstaller)
You can find prebuilt executables in the releases page instead of going through the source installation https://github.com/cromachina/art-timelapse/releases. It may still depend on ffmpeg (https://ffmpeg.org/download.html) being installed separately and added to the exe directory. One caveat of using the prebuilt pyinstaller executable is that it has a pretty slow startup time.

### Installing from source on Windows
- This method starts faster than the pyinstaller executable.
- Install python: https://www.python.org/downloads/ (add to your system PATH)
  - Alternatively: `winget install python`
- Install ffmpeg: https://ffmpeg.org/download.html
  - Alternatively: `winget install ffmpeg`
- With virtual environment:
  - Download the source to some location.
  - In the source directory, create a virtual environment: `python -m venv --system-site-packages venv`
  - Activate the virtual environment: `venv/bin/activate`
  - Install the project: `pip install -e .`
  - `art-timelapse` is now available in the current shell. You have to reactivate the venv if you open a new shell.
- Global install:
  - `pip install https://github.com/cromachina/art-timelapse/archive/refs/heads/main.zip`
  - The Python `Scripts` directory must be in your environment variables PATH to be able to run from any shell, for example:
    - `C:\Users\<Your-Username>\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts`
- Run with: `art-timelapse`
- This can be done similarly on Linux and MacOS, but you should use the Nix method instead.

### Installing with Nix on Ubuntu (or other Linux distros)
- Initial nix setup:
  - Install nix (may differ for your distro): `sudo apt install nix-bin`
  - Add yourself to the `nix-users` group: ``sudo usermod -a -G nix-users `whoami` ``
  - Refresh the group for your session: `newgrp nix-users`
  - Update `nix.conf` so you can use flakes:
    - `echo "extra-experimental-features = nix-command flakes" | sudo tee -a /etc/nix/nix.conf`
- Run: `nix run github:cromachina/art-timelapse`
- Run a specific version: `nix run github:cromachina/art-timelapse/v2.0.0`
- Optional: Consider using `system-manager` if you want to install in a persistent config file: https://github.com/numtide/system-manager

### Installing on NixOS
- Make sure experimental features are enabled for `nix-command` and `flakes`.
- Example, run from source directory: `nix run .#`
- Example, run from git: `nix run github:cromachina/art-timelapse`
- Example, add to your `configuration.nix` so you can run `art-timelapse` directly:
```nix
environment.systemPackages = with pkgs; [
  (builtins.getFlake "github:cromachina/art-timelapse/v2.0.0").packages.${pkgs.system}.default
];
```

### Running in Wayland
The dependencies `mss` (used for taking screenshots for both the window grabber and screen recording) and `pywinctl` (used by the window grabber and input tracker to identify the target window) do not currently work with Wayland compositors.
One way to get around this is to use Xwayland to run a lightweight X11 window manager, like i3, and then run both SAI and art-timelapse inside of that.

This is an example script that will start i3 in an Xwayland window, and then start SAI (configured with Bottles) and art-timelapse.
This script depends on: `xwayland-run`, `i3`, and `bottles`.
Change the `SAI_BOTTLE` and `SAI_PROGRAM` names according to your own Bottles config.
```sh
#!/usr/bin/env sh
SAI_BOTTLE=SAIv2
SAI_PROGRAM=sai2
export XDG_SESSION_TYPE=x11
export I3SOCK=~/.i3/i3-ipc.sock
cd ~
xwayland-run -decorate -- i3 &
until [ -e $I3SOCK ]; do sleep 1; done
i3-msg exec "workspace 1; exec bottles-cli run -b $SAI_BOTTLE -p $SAI_PROGRAM; exec art-timelapse"
```

Clipboard will also probably not work between the host and Xwayland windows. I don't know of a workaround for this.

### Todo
- Linux AppImage

https://github.com/cromachina/art-timelapse/assets/82557197/3e10a9d4-d855-4e91-8070-8f21aa9c350c
