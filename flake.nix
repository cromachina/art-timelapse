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
          version = "2.0.1";
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/96/e8/31450ac98740c52d553c9665979fa9f4a3dfedbc4e20b5650c8b54ea259b/pymemoryeditor-2.0.1-py3-none-any.whl";
            sha256 = "132427cf1762b9f0143a4c47c8e91968ff3d7746e7e39d0ee8e7ef9b4115fda7";
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
