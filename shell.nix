{ pkgs ? import <nixpkgs> { } }:
(pkgs.buildFHSEnv {
  name = "python-env";
  targetPkgs = pkgs: (with pkgs; [
    ffmpeg
    glib
    libGL
    poetry
    python313Full
    which
    xorg.libX11
    zlib
  ]);
}).env