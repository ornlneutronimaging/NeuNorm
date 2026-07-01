#!/usr/bin/env bash
#
# Launch the ORNL-styled NeuNorm Streamlit app and open it in Firefox.
#
# Thin wrapper around launch.sh so the two launchers share one implementation.
#
# Usage:
#   apps/streamlit/launch_ornl.sh            # default port 8501
#   PORT=8600 apps/streamlit/launch_ornl.sh  # override the port
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/launch.sh" neunorm_app_ornl_design.py
