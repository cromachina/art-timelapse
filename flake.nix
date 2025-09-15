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
          version = "1.5.23";
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/f9/a3/45b6a5c0fc7148752c2959bbfb4c964fbd2b95d36701e04cb5ca87e2a903/pymemoryeditor-1.5.23-py3-none-any.whl";
            sha256 = "1kv5kdzwls3p2pspsfylpq1qnazh879yc3qsng1nhqnpy5z9vhax";
          };
          format = "wheel";
          doCheck = false;
          buildInputs = [];
          checkInputs = [];
          nativeBuildInputs = [];
          propagatedBuildInputs = with pyPkgs; [
            psutil
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
