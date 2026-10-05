#!/usr/bin/env bash
# NVIDIA MPS (Multi-Process Service) control for the local GPU.
#
# Without MPS, the GPU gives each CUDA process a time slice, and the kernels of two queue lanes never run at the
# same time. With MPS, one server process runs the work of all clients, and their kernels can run at the same time.
#
# The daemon uses the default pipe folder /tmp/nvidia-mps. A new CUDA process of this user finds it and connects
# automatically: the queue scripts need no change. A process that started before the daemon stays a normal process.
# The GPU stays in the Default compute mode, so normal processes and the MPS server can use the GPU together.
#
# Usage: scripts/mps.sh start|status|stop
#   start   start the control daemon (the daemon starts the server when the first client connects)
#   status  show the daemon, the servers, the clients and the end of the control log
#   stop    stop the daemon; refuses while a CUDA client runs (the daemon then waits for the clients to end)

set -euo pipefail

export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/data/logs/nvidia-mps}

daemon_runs() {  # pgrep -x cannot match: Linux cuts process names to 15 characters
    local pid_file=$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid
    [[ -f $pid_file ]] && kill -0 "$(cat "$pid_file")" 2> /dev/null
}

case ${1:-status} in
    start)
        if daemon_runs; then
            echo "mps: the control daemon runs already"
            exit 0
        fi
        mkdir -p "$CUDA_MPS_LOG_DIRECTORY"
        nvidia-cuda-mps-control -d
        sleep 1
        daemon_runs || { echo "mps: the daemon did not start; see $CUDA_MPS_LOG_DIRECTORY/control.log" >&2; exit 1; }
        echo "mps: daemon started (pipe $CUDA_MPS_PIPE_DIRECTORY, log $CUDA_MPS_LOG_DIRECTORY)"
        ;;
    status)
        pgrep -a -f nvidia-cuda-mps || echo "mps: no daemon and no server"
        if daemon_runs; then
            servers=$(echo get_server_list | nvidia-cuda-mps-control)
            echo "servers: ${servers:-none}"
            for server in $servers; do
                echo "clients of $server: $(echo "get_client_list $server" | nvidia-cuda-mps-control | tr '\n' ' ')"
            done
        fi
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
        tail -n 5 "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>/dev/null || true
        ;;
    stop)
        daemon_runs || { echo "mps: no daemon"; exit 0; }
        clients=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v nvidia-cuda-mps-server || true)
        if [[ -n $clients ]]; then
            echo "mps: CUDA processes run; stop the daemon only between runs:" >&2
            echo "$clients" >&2
            exit 1
        fi
        echo quit | nvidia-cuda-mps-control
        echo "mps: daemon stopped"
        ;;
    *)
        echo "usage: scripts/mps.sh start|status|stop" >&2
        exit 2
        ;;
esac
