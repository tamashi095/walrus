{ pkgs, toolchain, packages }:
let
  specificRust = pkgs.rust-bin.fromRustupToolchainFile toolchain;
in
{
  core = pkgs.mkShell ({
    name = "core";
    buildInputs = [ specificRust ];
    DEV_SHELL_NAME = "walrus#core";
  });

  dev = pkgs.mkShell ({
    buildInputs = with pkgs; [
      specificRust
      openssl
    ];
    nativeBuildInputs = with pkgs; [
      clang
      libclang.lib
      llvmPackages.libcxxClang
      pkg-config
    ];
    DEV_SHELL_NAME = "walrus#dev";
    LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
    BINDGEN_EXTRA_CLANG_ARGS = "-isystem ${pkgs.llvmPackages.libcxxClang}/resource-root/lib/";
  });

  default = pkgs.mkShell ({
    buildInputs = with packages; [ walrus ];
    DEV_SHELL_NAME = "walrus#default";
  });
}
