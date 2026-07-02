{
    description = "fast-reform — per-point loop-closure warp for point clouds";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, flake-utils }:
        flake-utils.lib.eachDefaultSystem (system:
            let
                pkgs = import nixpkgs { inherit system; };
                rustToolchain = pkgs.rust-bin or null;
            in
            {
                devShells.default = pkgs.mkShell {
                    packages = [
                        pkgs.rustc
                        pkgs.cargo
                        pkgs.clippy
                        pkgs.rustfmt
                        pkgs.deno
                    ];

                    # The wasm target is added per-user with rustup; if you use the
                    # nixpkgs rustc above, add the wasm32-unknown-unknown target via
                    # your rust toolchain of choice.
                    shellHook = ''
                        echo "fast-reform dev shell"
                        echo "  ./run/build   compile native lib + wasm"
                        echo "  ./run/web     rebuild wasm, serve web/, open demo"
                        echo "  cargo test    run the suite"
                    '';
                };
            });
}
