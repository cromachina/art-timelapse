{ pkgs ? import <nixpkgs> { } }:
(pkgs.buildFHSEnv {
  name = "python-env";
  targetPkgs = pkgs: (with pkgs; [
    python313
    libGL
    glib
    xorg.libX11
    poetry
    linuxHeaders
    ffmpeg
    zlib
  ]);
}).env
