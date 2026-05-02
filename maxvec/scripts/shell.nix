let
  pkgs = import <nixpkgs> {};
in pkgs.mkShellNoCC {
  buildInputs = [
    (pkgs.arrow-cpp.overrideAttrs (oldAttrs: {
      cmakeFlags = (oldAttrs.cmakeFlags or []) ++ [ "-DARROW_ACERO=ON" ];
    }))
  ];

  packages = with pkgs; [
    clang-tools
    gcc
    gdb
    libcxx
    libcxxStdenv
    cmake
    ninja
    mold

    taskflow

    python312
    python312Packages.duckdb
    python312Packages.pyarrow
  ];

  TESTLD = pkgs.mold;
  shellHook = ''
      export LD=${pkgs.mold}/bin/mold
  '';
}
