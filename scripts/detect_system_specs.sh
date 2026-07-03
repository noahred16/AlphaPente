#!/usr/bin/env bash
# Detects system CPU and memory and writes suggested .env values (80% of available).
# Run from anywhere in the repo; writes to <repo-root>/.env.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

mem_from_system() {
  if [[ -f /proc/meminfo ]]; then
    echo $(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
  else
    echo $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
  fi
}

# Prefer cgroup v2 CPU quota over nproc, which sees the full host on shared servers.
CGROUP_CPU_MAX=/sys/fs/cgroup/cpu.max
if [[ -r "$CGROUP_CPU_MAX" ]]; then
  read -r quota period < "$CGROUP_CPU_MAX"
  if [[ "$quota" != "max" && "$period" -gt 0 ]]; then
    TOTAL_THREADS=$(echo "$quota $period" | awk '{printf "%d", $1/$2}')
    [[ "$TOTAL_THREADS" -lt 1 ]] && TOTAL_THREADS=1
  else
    TOTAL_THREADS=$(nproc)
  fi
else
  if command -v nproc &>/dev/null; then
    TOTAL_THREADS=$(nproc)
  else
    TOTAL_THREADS=$(sysctl -n hw.logicalcpu)
  fi
fi
NUM_THREADS=$(( TOTAL_THREADS * 8 / 10 ))
[ "$NUM_THREADS" -lt 1 ] && NUM_THREADS=1

# Prefer cgroup v2 memory limit over /proc/meminfo, which sees the full host on shared servers.
CGROUP_MEM_MAX=/sys/fs/cgroup/memory.max
if [[ -r "$CGROUP_MEM_MAX" ]]; then
  MEM_LIMIT_BYTES=$(cat "$CGROUP_MEM_MAX")
  if [[ "$MEM_LIMIT_BYTES" != "max" ]]; then
    TOTAL_MEM_GB=$(( MEM_LIMIT_BYTES / 1024 / 1024 / 1024 ))
  else
    TOTAL_MEM_GB=$(mem_from_system)
  fi
else
  TOTAL_MEM_GB=$(mem_from_system)
fi
ARENA_SIZE_GB=$(( TOTAL_MEM_GB * 8 / 10 ))
[ "$ARENA_SIZE_GB" -lt 1 ] && ARENA_SIZE_GB=1

echo "Detected system specs:"
echo "  CPU threads : $TOTAL_THREADS total → NUM_THREADS=$NUM_THREADS (80%)"
echo "  RAM         : ${TOTAL_MEM_GB} GB total → ARENA_SIZE_GB=${ARENA_SIZE_GB} GB (80%)"
echo ""

cat > "$ENV_FILE" <<EOF
ARENA_SIZE_GB=$ARENA_SIZE_GB
NUM_THREADS=$NUM_THREADS
EOF

echo "Written to $ENV_FILE"
