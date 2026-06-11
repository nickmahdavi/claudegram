{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = [
    pkgs.python312
    pkgs.uv
  ];

  shellHook = ''
    # Use the nix-provided Python rather than having uv download its own.
    export UV_PYTHON="${pkgs.python312}/bin/python3"
  '';
}
