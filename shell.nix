{ pkgs ? import <nixpkgs> { } }:
(pkgs.buildFHSEnv {
  name = "python-env";
  targetPkgs = pkgs: (with pkgs; [
    ffmpeg
    glib
    libGL
    python313Full
    which
    xorg.libX11
    zlib
    linuxHeaders
  ]);
  runScript = pkgs.writeScript "init.sh" ''
    python -m venv venv
    source venv/bin/activate
    exec bash
  '';
}).env