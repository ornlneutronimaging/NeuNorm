#!/usr/bin/env bash
#
# Launch the NeuNorm Streamlit app and open it in Firefox.
#
# Starts the Streamlit server (headless, via the Pixi `streamlit` environment),
# waits for it to become healthy, opens Firefox on the app URL, then keeps the
# server running until you press Ctrl-C (which shuts the server down cleanly).
#
# Usage:
#   apps/streamlit/launch.sh                             # base app, port 8501
#   apps/streamlit/launch.sh neunorm_app_ornl_design.py  # ORNL-styled app
#   PORT=8600 apps/streamlit/launch.sh                   # override the port
#
set -euo pipefail

# --- Resolve paths (works regardless of the current working directory) ------ #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# App to launch: first argument (bare name or path), defaults to the base app.
APP_ARG="${1:-neunorm_app.py}"
if [[ "${APP_ARG}" = /* ]]; then
    APP="${APP_ARG}"                 # absolute path as-given
else
    APP="${SCRIPT_DIR}/${APP_ARG}"   # resolve relative to this script's dir
fi
if [[ ! -f "${APP}" ]]; then
    echo "ERROR: app file not found: ${APP}" >&2
    exit 1
fi

# --- Pick a free port (start at $PORT, default 8501; scan up to +20) -------- #
START_PORT="${PORT:-8501}"
PORT=""
for p in $(seq "${START_PORT}" "$((START_PORT + 20))"); do
    if ! { exec 3<>"/dev/tcp/127.0.0.1/${p}"; } 2>/dev/null; then
        PORT="${p}"
        break
    fi
    exec 3>&- 2>/dev/null || true
done
if [[ -z "${PORT}" ]]; then
    echo "ERROR: no free port found in ${START_PORT}..$((START_PORT + 20))" >&2
    exit 1
fi

URL="http://localhost:${PORT}"

# --- Skip Streamlit's one-time interactive "email" prompt ------------------- #
# (it blocks on stdin and would hang a non-interactive launch)
CRED="${HOME}/.streamlit/credentials.toml"
if [[ ! -f "${CRED}" ]]; then
    mkdir -p "${HOME}/.streamlit"
    printf '[general]\nemail = ""\n' >"${CRED}"
fi

# --- Start the server ------------------------------------------------------- #
echo "Starting NeuNorm Streamlit app on ${URL} ..."
cd "${REPO_ROOT}"
pixi run -e streamlit streamlit run "${APP}" \
    --server.headless true \
    --server.port "${PORT}" \
    --browser.gatherUsageStats false &
SERVER_PID=$!

# Always stop the server when this script exits (Ctrl-C, error, or normal end).
cleanup() {
    echo
    echo "Stopping NeuNorm Streamlit app (pid ${SERVER_PID}) ..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- Wait until the server answers its health check ------------------------- #
echo -n "Waiting for the server to come up"
for _ in $(seq 1 60); do
    if curl -fsS -o /dev/null "${URL}/_stcore/health" 2>/dev/null; then
        echo " ready."
        break
    fi
    # Bail out early if the server process already died.
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo
        echo "ERROR: the Streamlit server exited before becoming ready." >&2
        exit 1
    fi
    echo -n "."
    sleep 1
done

# --- Open Firefox ----------------------------------------------------------- #
if command -v firefox >/dev/null 2>&1; then
    echo "Opening Firefox at ${URL}"
    firefox --new-window "${URL}" >/dev/null 2>&1 &
else
    echo "WARNING: firefox not found on PATH; open ${URL} manually." >&2
fi

echo
echo "NeuNorm app is running at ${URL}"
echo "Press Ctrl-C to stop."

# Keep running (and keep the trap armed) until the server stops or Ctrl-C.
wait "${SERVER_PID}"
