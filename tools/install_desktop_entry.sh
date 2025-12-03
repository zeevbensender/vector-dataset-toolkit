#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ICON_SOURCE="${REPO_ROOT}/resources/icons/vector_dataset_tool_256x256.png"
DESKTOP_FILE="${REPO_ROOT}/resources/vector-dataset-tool.desktop"

APPLICATIONS_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/applications"
ICON_TARGET_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/icons/hicolor/256x256/apps"
DESKTOP_TARGET="${APPLICATIONS_DIR}/vector-dataset-tool.desktop"
ICON_TARGET="${ICON_TARGET_DIR}/vector_dataset_tool.png"

if [[ ! -f "${ICON_SOURCE}" ]]; then
    echo "Icon not found at ${ICON_SOURCE}. Run tools/generate_icons.py first." >&2
    exit 1
fi

install -Dm644 "${DESKTOP_FILE}" "${DESKTOP_TARGET}"
install -Dm644 "${ICON_SOURCE}" "${ICON_TARGET}"

echo "Installed desktop entry to ${DESKTOP_TARGET}"
echo "Installed icon to ${ICON_TARGET}"

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "${APPLICATIONS_DIR}" || true
fi
