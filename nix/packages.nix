{ pkgs, toolchain, self, crane }:
let
  specificRust = pkgs.rust-bin.fromRustupToolchainFile toolchain;
  craneLib = (crane.mkLib pkgs).overrideToolchain (p: specificRust);
  cargoToml = "${self}/Cargo.toml";
  cargoTomlConfig = builtins.fromTOML (builtins.readFile cargoToml);
  version = cargoTomlConfig.workspace.package.version;
  src = self;
  buildInputs = with pkgs; [
    clang
    libclang.lib
    llvmPackages.libcxxClang
    openssl
  ];
  nativeBuildInputs = with pkgs; [ pkg-config ];
  outputHashes = {

  };
  doCheck = false;
  env = {
    GIT_REVISION="unstable";
    LIBCLANG_PATH = "${pkgs.libclang.lib}/lib";
    BINDGEN_EXTRA_CLANG_ARGS = "-isystem ${pkgs.llvmPackages.libcxxClang}/resource-root/lib/";
  };
in
rec {
  default = walrus;

  walrus = craneLib.buildPackage {
    inherit version src env cargoToml buildInputs nativeBuildInputs outputHashes doCheck;
    pname = "walrus";
    cargoExtraArgs = "--bin walrus";
    cargoArtifacts = craneLib.buildDepsOnly {
      inherit version src env cargoToml buildInputs nativeBuildInputs outputHashes doCheck;
      pname = "walrus";
      cargoExtraArgs  = "--bin walrus";
    };
  };
  walursVersionRust = specificRust;
}
