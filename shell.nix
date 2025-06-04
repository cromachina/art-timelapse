{ pkgs ? import <nixpkgs> { } }:
let
  pythonldlibpath = pkgs.lib.makeLibraryPath (with pkgs; [
    stdenv.cc.cc
    libGL
    glib
  ]);
in
pkgs.mkShell {
  packages = with pkgs; [
    ffmpeg
    python313Full
    xorg.libX11
    zlib
    linuxHeaders
  ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pythonldlibpath}
    python -m venv venv
    source venv/bin/activate
    exec bash
  '';
}