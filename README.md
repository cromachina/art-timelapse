# Art Timelapse
This program lets you create efficient timelapses of your drawing.

### Why
I made this program to be used primarily with Paint Tool SAIv2, (which does not have any builtin timelapse tool), however it can be used with any other program.
Previously I was reluctant to record drawing videos with OBS Studio, because I would have to do post-processing on every video:
Cutting dead air from the videos; Concatenating separate recordings together; Speeding up the resulting video to make a timelapse.
While it is possible to mitigate these issues when using OBS Studio, I wanted a more accurate solution that works sort of like Clip Studio's timelapse feature.
By recording the end of each stroke of a drawing, you get the largest amount of information about the timelapse without any redundancy, which can end up creating compact videos with minimal editing.

### Corrupt video recovery after system crash
If you have a system crash or power failure while you were recording, the resulting video may be truncated and have missing header information.
Some frames may be lost if they were not flushed to disk before the crash occurred. Anything that was written to disk may still be recoverable.
You first need to make a reference video file from which to restore header information, so create a new short recording, perferably with the same dimensions as the corrupt video.
Then use a recovery tool that can use the reference video to fix the corrupt video, such as https://github.com/anthwlock/untrunc. This may also be done with FFMPEG https://www.handyrecovery.com/fix-corrupted-video-using-ffmpeg/.
In order to mitigate the chance that a recorded video is completely unrecoverable, you can configure a setting in this tool to automatically split recorded output after some number of frames, so that smaller video files are completely written more frequently.

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
- Run a specific version: `nix run github:cromachina/art-timelapse/v3.0.0`
- Optional: Consider using `system-manager` if you want to install in a persistent config file: https://github.com/numtide/system-manager

### Installing on NixOS
- Make sure experimental features are enabled for `nix-command` and `flakes`.
- Example, run from source directory: `nix run`
- Example, run from git: `nix run github:cromachina/art-timelapse`
- Example, add to your `configuration.nix` so you can run `art-timelapse` directly:
```nix
environment.systemPackages = with pkgs; [
  (builtins.getFlake "github:cromachina/art-timelapse/v3.0.0").packages.${pkgs.system}.default
];
```

### Local dev
- Run `python build_local.py` to build or update translation files for local testing.

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

### Dev notes:
#### Procedure for locating the canvas storage pointer in a different SAIv2 version.
The goal is the find a route from the canvas storage pointer to the canvas composite data.
1. Open SAI.
2. Open canvas with a color pattern in the upper left corner, for example:
   Paint randomly colored pixels in the format of `BGRA` like the following (`A` will always be `00`).
   ```
   FF 30 00 00 00 46 FF 00 00 FF 91 00
   ```
3. Open Cheat Engine and attach to SAI.
4. Run a scan for the byte array pattern above. You are looking for the pattern at a page aligned address, like `xxxx0000`. If you use at least 3 pixels (12 bytes), you'll usually end up with a single result. This address is the canvas composite data's first block.
5. Save that found address to the address list.
6. Create a new pointer map and save it (like pmap1.scandata).
7. Save the canvas and close it, then reopen it.
8. Repeat steps 4 to 7 while saving a new pointer map each time, at least 3 times total, however leave the last canvas open after saving the last map.
9. Right click on the last address saved and run "Pointer scan for this address".
10. Select "Use saved pointermap" and select the last pointer map saved.
11. Select "Compare results with other saved pointermaps"
12. Add each successive pointer map and its associated address.
13. Select other options:
 - Only find paths with a static address
 - Don't include pointers with read-only nodes
 - No looping pointers
 - Max different offsets per node: 3
 - Maximum offset value: 4095
 - Max level: 7
14. Click okay to run the scan. Hopefully a very short list appears, with maybe 20 to 30 entries.
15. The address with a `0` in the `Offset 0` column is most likely the target. This may not always be true for every version, but it greatly narrows down the potential pointers for further analysis.

Unless there is a major refactor, the layouts of structures do not seem to change much, so often adding support for another version is just searching for the canvas storage pointer location and deriving a new SAI API class from the previous version.
