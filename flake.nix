{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
  flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      lib = pkgs.lib;
      python = pkgs.python313;
      pyPkgs = python.pkgs // {
        pymemoryeditor = pyPkgs.buildPythonPackage {
          pname = "pymemoryeditor";
          version = "1.5.24";
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/e5/0f/efcf707643bdce3831726a55361b72754788d8b8502a48bdf7005b8c214c/pymemoryeditor-1.5.24-py3-none-any.whl";
            sha256 = "0d607bd43d49fa7b5c8479d9c53ea2386baf06ab298b9c464bb03553babbd60e";
          };
          format = "wheel";
          doCheck = false;
          propagatedBuildInputs = with pyPkgs; [
            psutil
          ];
        };
        ttkbootstrap = pyPkgs.buildPythonPackage rec {
          pname = "ttkbootstrap";
          version = "1.20.1";
          src = pkgs.fetchFromGitHub {
            owner = "israel-dryer";
            repo = "ttkbootstrap";
            rev = "v${version}";
            hash = "sha256-HZiLd134J0I+4egpLjTO+b+ZSyRjBSH8bf69jLOjAh0=";
          };
          format = "pyproject";
          doCheck = false;
          build-system = with pyPkgs; [
            setuptools
          ];
          propagatedBuildInputs = with pyPkgs; [
            pillow
          ];
        };
      };
      pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
      project = pyproject.project;
      fixString = x: lib.strings.toLower (builtins.replaceStrings ["_"] ["-"] x);
      getPkgs = x: lib.attrsets.attrVals (builtins.map fixString x) pyPkgs;
      package = pyPkgs.buildPythonPackage {
        pname = project.name;
        version = project.version;
        format = "pyproject";
        src = ./.;
        build-system = getPkgs pyproject.build-system.requires;
        dependencies = getPkgs project.dependencies ++ [
          pyPkgs.tkinter
          pkgs.ffmpeg-full
        ];
      };
      editablePackage = pyPkgs.mkPythonEditablePackage {
        pname = project.name;
        version = project.version;
        scripts = project.scripts;
        root = "$PWD/src";
        dependencies = getPkgs project.optional-dependencies.dev;
      };
    in
    {
      packages.default = pyPkgs.toPythonApplication package;
      devShells.default = pkgs.mkShell {
        inputsFrom = [
          package
        ];
        buildInputs = [
          editablePackage
          pyPkgs.build
        ];
      };
    }
  );
}
