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
      python = pkgs.python314;
      pyPkgs = python.pkgs // {
        pymemoryeditor = pyPkgs.buildPythonPackage {
          pname = "pymemoryeditor";
          version = "2.0.0";
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/72/ef/aba8f986b7e143c16394fd6ac19dd0b5fcb9934962830821063bca5c0b72/pymemoryeditor-2.0.0-py3-none-any.whl";
            sha256 = "efd464f08064c3547ce00090af878fb8a819796adbd6b29b11b5ccee6659b6b7";
          };
          format = "wheel";
          doCheck = false;
          propagatedBuildInputs = with pyPkgs; [
            psutil
          ];
        };
      };
      pyproject = fromTOML (builtins.readFile ./pyproject.toml);
      project = pyproject.project;
      fixString = x: lib.strings.toLower (builtins.replaceStrings ["_"] ["-"] x);
      getPkgs = x: lib.attrsets.attrVals (map fixString x) pyPkgs;
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
        dependencies = package.build-system ++ getPkgs project.optional-dependencies.dev;
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
