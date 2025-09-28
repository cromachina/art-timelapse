import logging
import argparse
import asyncio

from . import timelapse, gui

async def async_main():
    logging.basicConfig(
        format='[%(asctime)s][%(levelname)s] %(message)s',
        level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', action='store_true', help='Run as a CLI app instead of a GUI app.')
    parser.add_argument('--frames', help='Name of the file or directory to store frames or videos.')
    parser.add_argument('--sai', action='store_true', help='Read SAI\'s memory directly to capture frames. Prompts for canvas selection.')
    parser.add_argument('--psd-file', help='Instead of screen recording, record the specified PSD file every time it is written to.')
    parser.add_argument('--image-size-limit', type=int, default=1000, help='Limit the resolution from a PSD or SAI capture.')
    parser.add_argument('--export', action='store_true', help='Export the given frame data file to an MP4, or concatenate a video directory. Figures out what to do based on the frames path.')
    parser.add_argument('--export-time-limit', type=float, default=60, help='Compress the play time of the exported MP4 file to be no longer than the given seconds. Default is 60 seconds. Set to 0 for uncompressed play time.')
    parser.add_argument('--drag-grab', action='store_true', help='Drag a rectangle over a window to capture that area. Useful for when a subwindow cannot be captured by click.')
    parser.add_argument('--container', default='webm', help='The format to store video in, for example "webm", "mp4", "avi". Default for storage is webm with VP8 codec for better color quality, but it takes up more space than mp4 with avc1.')
    parser.add_argument('--codec', default='vp80', help='The fourcc for the video container\'s codec. Default for storage is VP8 (vp80).')
    parser.add_argument('--web', action='store_true', help='Sets the video container and codec to "mp4" and "avc1" respectively, which is what Twitter and many other websites expect for video uploads. Use this with --export.')
    config = parser.parse_args()
    if config.cli:
        await timelapse.main(config)
    else:
        await gui.main()

def main():
    asyncio.run(async_main())