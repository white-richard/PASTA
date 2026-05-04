#!/usr/bin/env bash
set -euo pipefail

monitor_cmd() {
  local label="$1"
  shift

  local output_dir="$1"
  shift

  local memlog_dir="${output_dir}/memlogs"
  mkdir -p "${memlog_dir}"

  local stdout_log="${memlog_dir}/${label}.stdout.log"
  local stderr_log="${memlog_dir}/${label}.stderr.log"
  local summary_log="${memlog_dir}/${label}.summary.txt"
  local pid_log="${memlog_dir}/${label}.pid"

  # Write directly to the log file instead of through a FIFO.  A FIFO
  # requires a live reader process; if the reader is killed (e.g. by
  # accelerate/torchrun killing the process group on worker failure) the
  # write end gets SIGPIPE and output is lost.  Direct >> is not
  # susceptible to this and survives process-group signals.
  (
    "$@" 2>&1
  ) >>"${stdout_log}" &
  local cmd_pid=$!

  echo "${cmd_pid}" >"${pid_log}"

  # Show live output on the terminal.  This process is expendable: if it
  # dies (SSH disconnect, terminal close) the training is unaffected.
  tail -f "${stdout_log}" &
  local tail_pid=$!

  local max_rss_kb=0
  local max_vram_mb=0

  while kill -0 "${cmd_pid}" 2>/dev/null; do
    local rss_kb=0
    rss_kb=$(ps -o rss= -p "${cmd_pid}" 2>/dev/null | awk '{print $1+0}')
    if [[ "${rss_kb}" -gt "${max_rss_kb}" ]]; then
      max_rss_kb="${rss_kb}"
    fi

    local vram_mb=0
    vram_mb=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print $1+0}')
    if [[ "${vram_mb}" -gt "${max_vram_mb}" ]]; then
      max_vram_mb="${vram_mb}"
    fi

    sleep 1
  done

  wait "${cmd_pid}"
  local exit_code=$?

  kill "${tail_pid}" 2>/dev/null || true
  wait "${tail_pid}" 2>/dev/null || true

  {
    echo "label=${label}"
    echo "pid=${cmd_pid}"
    echo "max_rss_kb=${max_rss_kb}"
    echo "max_rss_mb=$((max_rss_kb / 1024))"
    echo "max_vram_mb=${max_vram_mb}"
    echo "stdout_log=${stdout_log}"
    echo "stderr_log=${stderr_log}"
    echo "[${label}] max RAM: $((max_rss_kb / 1024)) MB"
    echo "[${label}] max VRAM: ${max_vram_mb} MB"
    echo "[${label}] logs saved to ${memlog_dir}"
  } >"${summary_log}"

  {
    echo "[${label}] max RAM: $((max_rss_kb / 1024)) MB"
    echo "[${label}] max VRAM: ${max_vram_mb} MB"
    echo "[${label}] logs saved to ${memlog_dir}"
  } >>"${stdout_log}"

  echo "[${label}] max RAM: $((max_rss_kb / 1024)) MB"
  echo "[${label}] max VRAM: ${max_vram_mb} MB"
  echo "[${label}] logs saved to ${memlog_dir}"

  return "${exit_code}"
}
