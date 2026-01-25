#!/bin/bash
# Ray debugging utilities for multi-node SLURM jobs
# Source this file after setting up Ray environment variables

# Print comprehensive Ray debug information
print_ray_debug_info() {
    local nodes=("$@")
    local head_node="${nodes[0]}"

    echo "===== RAY DEBUG BEGIN ====="
    echo "[ray-debug] date: $(date)"
    echo "[ray-debug] job_id: ${SLURM_JOB_ID:-}"
    echo "[ray-debug] nodes: ${nodes[*]}"
    echo "[ray-debug] head_node: ${head_node}"
    echo "[ray-debug] RAY_HEAD_IP env: ${RAY_HEAD_IP:-<unset>}"
    echo "[ray-debug] RAY_INTERFACE: ${RAY_INTERFACE:-<unset>}"
    echo "[ray-debug] RAY_PORT env: ${RAY_PORT:-<unset>}"
    echo "[ray-debug] NUM_GPUS_PER_NODE: ${NUM_GPUS_PER_NODE}"
    echo "[ray-debug] NUM_CPUS_PER_NODE: ${NUM_CPUS_PER_NODE}"
    echo "[ray-debug] VENV_DIR: ${VENV_DIR}"
    echo "[ray-debug] PATH: ${PATH}"
    echo "[ray-debug] which ray: $(command -v ray || echo '<not-found>')"
    echo "[ray-debug] ray --version: $(${RAY_CMD} --version 2>/dev/null || echo '<failed>')"
    echo "[ray-debug] python -V: $(${VENV_DIR}/bin/python -V 2>/dev/null || echo '<failed>')"
    echo "[ray-debug] NCCL_IB_DISABLE: ${NCCL_IB_DISABLE}"
    echo "[ray-debug] NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-<unset>}"
    echo "[ray-debug] NCCL_NET_GDR_LEVEL: ${NCCL_NET_GDR_LEVEL:-<unset>}"
    echo "[ray-debug] RAY_TMPDIR: ${RAY_TMPDIR}"
    echo "[ray-debug] RAY_LOG_DIR: ${RAY_LOG_DIR}"
    echo "[ray-debug] RAY_ADDRESS pre-set: ${RAY_ADDRESS:-<unset>}"

    for node in "${nodes[@]}"; do
        echo "[ray-debug] ---- node: ${node} ----"
        srun --nodes=1 --ntasks=1 -w "$node" /bin/bash -c \
            "echo '[ray-debug] hostname: ' \$(hostname); \
             echo '[ray-debug] hostname -I: ' \$(hostname -I); \
             command -v ip >/dev/null 2>&1 && ip -4 addr || true; \
             command -v ip >/dev/null 2>&1 && ip route | head -n 5 || true; \
             echo '[ray-debug] ray: ' \$(command -v ray || echo '<not-found>'); \
             echo '[ray-debug] ray --version: ' \$(${RAY_CMD} --version 2>/dev/null || echo '<failed>'); \
             echo '[ray-debug] python -V: ' \$(${VENV_DIR}/bin/python -V 2>/dev/null || echo '<failed>');" || true
    done
    echo "===== RAY DEBUG END ====="
}

# Dump Ray logs from a specific node
dump_ray_logs() {
    local node="$1"
    srun --nodes=1 --ntasks=1 -w "$node" /bin/bash -c \
        "logroot='${RAY_TMPDIR}'; \
         echo \"===== RAY LOG DUMP BEGIN (${node}) =====\"; \
         echo \"logroot: \${logroot}\"; \
         found_session=\"\"; \
         for root in \"${RAY_TMPDIR}/ray\" \"${RAY_TMPDIR}\" /tmp/ray; do \
           if [[ -d \"\${root}\" ]]; then \
             latest=\$(ls -dt \${root}/session_* 2>/dev/null | head -n1); \
             if [[ -n \"\${latest}\" ]]; then \
               echo \"session root: \${root}\"; \
               echo \"latest session: \${latest}\"; \
               found_session=\"\${latest}\"; \
               break; \
             fi; \
           fi; \
         done; \
         if [[ -n \"\${found_session}\" && -d \"\${found_session}/logs\" ]]; then \
           for f in raylet.err raylet.out gcs_server.err gcs_server.out monitor.err monitor.out dashboard.log; do \
             if [[ -f \"\${found_session}/logs/\${f}\" ]]; then \
               echo \"----- \${f} (tail 200) -----\"; \
               tail -n 200 \"\${found_session}/logs/\${f}\"; \
             else \
               echo \"----- \${f} missing -----\"; \
             fi; \
           done; \
         else \
           echo \"no session logs found under ${RAY_TMPDIR} or /tmp/ray\"; \
         fi; \
         if [[ -d '${RAY_LOG_DIR}' ]]; then \
           echo \"----- srun logs for ${node} (tail 200) -----\"; \
           for f in '${RAY_LOG_DIR}'/ray_*_${node}_*.out '${RAY_LOG_DIR}'/ray_*_${node}_*.err '${RAY_LOG_DIR}'/ray_head_*.out '${RAY_LOG_DIR}'/ray_head_*.err; do \
             if [[ -f \"\${f}\" ]]; then \
               echo \"----- \${f} -----\"; \
               tail -n 200 \"\${f}\"; \
             fi; \
           done; \
         fi; \
         echo \"===== RAY LOG DUMP END (${node}) =====\";" || true
}

# Dump logs from all nodes
dump_all_ray_logs() {
    local nodes=("$@")
    echo "===== RAY LOG DUMP (ALL NODES) ====="
    for node in "${nodes[@]}"; do
        dump_ray_logs "$node"
    done
    echo "===== RAY LOG DUMP COMPLETE ====="
}

# Wait for all nodes to join the Ray cluster
wait_for_ray_cluster() {
    local expected_nodes="$1"
    shift
    local nodes=("$@")

    local active_nodes=0
    echo "===== RAY JOIN CHECK BEGIN ====="
    for i in $(seq 1 "${RAY_JOIN_MAX_ATTEMPTS}"); do
        active_nodes=$(
            ${RAY_CMD} status --address "${RAY_ADDRESS}" 2>/dev/null | awk '
                /^Active:/ {in_active=1; next}
                /^Pending:/ {in_active=0}
                in_active && /node_/ {count++}
                END {print count+0}
            '
        ) || true
        active_nodes=$(printf "%s" "${active_nodes}" | tail -n1 | tr -cd '0-9')
        active_nodes="${active_nodes:-0}"
        echo "[ray-join] active_nodes=${active_nodes} expected=${expected_nodes}"
        if [[ "${active_nodes}" =~ ^[0-9]+$ ]] && [[ "${active_nodes}" -ge "${expected_nodes}" ]]; then
            break
        fi
        sleep "${RAY_JOIN_SLEEP}"
    done

    if [[ "${active_nodes}" -lt "${expected_nodes}" ]]; then
        echo "[ray-join] ERROR: only ${active_nodes}/${expected_nodes} nodes joined; dumping logs."
        dump_all_ray_logs "${nodes[@]}"
        echo "===== RAY JOIN CHECK END ====="
        return 1
    fi
    echo "===== RAY JOIN CHECK END ====="
    return 0
}

# Print current Ray cluster status
print_ray_status() {
    echo "===== RAY STATUS BEGIN ====="
    ${RAY_CMD} status --address "${RAY_ADDRESS}" || true
    echo "===== RAY STATUS END ====="
}
