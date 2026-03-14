#!/usr/bin/env bash
#
# install_blender.sh — Automated Blender installation (non-APT)
# Tested on Ubuntu 22.04 / 24.04
#
# Installs Blender from the official Blender Foundation tarballs.
# Designed for clean system-wide installs under /opt/blender.
#

set -euo pipefail
IFS=$'\n\t'

# ---- CONFIGURATION ----
BLENDER_VERSION_DEFAULT="4.2.3"  # change to preferred version
INSTALL_DIR="/opt/blender"
BIN_SYMLINK="/usr/local/bin/blender"
DESKTOP_ENTRY="/usr/share/applications/blender.desktop"
TMP_DIR="/tmp/blender-install.$$"
BLENDER_URL_BASE="https://download.blender.org/release/Blender"

# ---- FUNCTIONS ----

log() { echo -e "\033[1;32m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERR]\033[0m $*" >&2; exit 1; }

check_prerequisites() {
    command -v wget >/dev/null 2>&1 || err "wget not found. Install it first."
    command -v tar >/dev/null 2>&1 || err "tar not found. Install it first."
    command -v sha256sum >/dev/null 2>&1 || err "sha256sum not found. Install it first."
}

prompt_version() {
    read -rp "Enter Blender version to install [${BLENDER_VERSION_DEFAULT}]: " v
    BLENDER_VERSION="${v:-$BLENDER_VERSION_DEFAULT}"
    BLENDER_MAJOR_MINOR="$(echo "$BLENDER_VERSION" | cut -d. -f1,2)"
    BLENDER_TARBALL="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
    BLENDER_URL="${BLENDER_URL_BASE}${BLENDER_MAJOR_MINOR}/${BLENDER_TARBALL}"
}

download_blender() {
    mkdir -p "$TMP_DIR"
    log "Downloading Blender ${BLENDER_VERSION}..."
    wget -q --show-progress -O "$TMP_DIR/${BLENDER_TARBALL}" "$BLENDER_URL" || \
        err "Failed to download Blender tarball."
}

install_blender() {
    log "Extracting Blender to ${INSTALL_DIR}/${BLENDER_VERSION}..."
    sudo mkdir -p "$INSTALL_DIR"
    sudo tar -xf "$TMP_DIR/${BLENDER_TARBALL}" -C "$INSTALL_DIR"
    sudo mv "$INSTALL_DIR/blender-${BLENDER_VERSION}-linux-x64" "$INSTALL_DIR/${BLENDER_VERSION}"

    log "Creating symlink ${BIN_SYMLINK}..."
    sudo ln -sf "${INSTALL_DIR}/${BLENDER_VERSION}/blender" "$BIN_SYMLINK"
}

create_desktop_entry() {
    log "Creating desktop shortcut..."
    sudo tee "$DESKTOP_ENTRY" >/dev/null <<EOF
[Desktop Entry]
Name=Blender ${BLENDER_VERSION}
GenericName=3D Creation Suite
Comment=Blender is the free and open source 3D creation suite.
Exec=${BIN_SYMLINK} %f
Icon=${INSTALL_DIR}/${BLENDER_VERSION}/blender.svg
Terminal=false
Type=Application
Categories=Graphics;3DGraphics;
MimeType=application/x-blender;
EOF
}

cleanup() {
    rm -rf "$TMP_DIR"
}

verify_install() {
    log "Verifying Blender installation..."
    if ! blender -v >/dev/null 2>&1; then
        err "Blender binary not found or not executable."
    fi
    log "Blender installed successfully → $(blender -v)"
}

# ---- MAIN EXECUTION ----
check_prerequisites
prompt_version
download_blender
install_blender
create_desktop_entry
cleanup
verify_install
