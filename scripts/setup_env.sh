#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="$ROOT/.envs/pgsr-1.0.0"
CACHE_ROOT="$ROOT/.cache"
CONDA_BIN="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
FSUTIL_BIN="$(command -v fsutil.exe || true)"
CONSTRAINTS="$ROOT/scripts/setup_env_constraints.txt"
GSPLAT_WHEEL='https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%2Bpt24cu121-cp310-cp310-linux_x86_64.whl'

mkdir -p "$ROOT/.envs" "$CACHE_ROOT/conda/pkgs" "$CACHE_ROOT/pip" \
    "$CACHE_ROOT/tmp" "$CACHE_ROOT/torch_extensions" "$CACHE_ROOT/triton" \
    "$CACHE_ROOT/cuda" "$CACHE_ROOT/xdg" "$CACHE_ROOT/matplotlib" \
    "$CACHE_ROOT/torch" "$CACHE_ROOT/opencv"

export CONDA_PKGS_DIRS="$CACHE_ROOT/conda/pkgs"
export CONDA_ENVS_PATH="$ROOT/.envs"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TMPDIR="$CACHE_ROOT/tmp"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export TORCH_HOME="$CACHE_ROOT/torch"
export OPENCV_OPENCL_CACHE_DIR="$CACHE_ROOT/opencv"

LOG="$CACHE_ROOT/setup_env_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -x "$CONDA_BIN" ]] || fail "Miniconda not executable at $CONDA_BIN"
[[ -f "$CONSTRAINTS" ]] || fail "Missing constraints file: $CONSTRAINTS"

if [[ ! -r /etc/os-release ]]; then
    fail 'Cannot identify the Linux distribution.'
fi
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 24.04 ]] || \
    fail "Expected Ubuntu 24.04; found ${PRETTY_NAME:-unknown}."
[[ "$(uname -r)" == *microsoft* ]] || fail 'Expected a WSL2 Linux kernel.'

AVAILABLE_GIB="$(df -BG --output=avail "$ROOT" | tail -n 1 | tr -cd '0-9')"
[[ -n "$AVAILABLE_GIB" ]] || fail 'Could not determine free space on the workspace drive.'
(( AVAILABLE_GIB >= 30 )) || fail "Only ${AVAILABLE_GIB} GiB free on the workspace drive; need at least 30 GiB."
check_disk_budget() {
    local stage="$1"
    local e_bytes c_bytes
    e_bytes="$(df -B1 --output=avail "$ROOT" | tail -n 1 | tr -cd '0-9')"
    c_bytes="$(df -B1 --output=avail /mnt/c 2>/dev/null | tail -n 1 | tr -cd '0-9' || true)"
    [[ -n "$e_bytes" ]] || fail "Cannot recheck E: free space at $stage."
    (( e_bytes >= 30 * 1024 * 1024 * 1024 )) || \
        fail "E: fell below 30 GiB at $stage."
    if [[ -n "$c_bytes" ]]; then
        (( c_bytes >= 20000000000 )) || \
            fail "C: fell below 20,000,000,000 bytes at $stage."
        printf 'Disk space at %s: E=%s bytes C=%s bytes\n' "$stage" "$e_bytes" "$c_bytes"
    else
        printf 'Disk space at %s: E=%s bytes; C: is not mounted in WSL.\n' "$stage" "$e_bytes"
    fi
}

case_sensitive_preflight() {
    local directory="$1"
    local windows_path probe upper_inode lower_inode
    [[ -n "$FSUTIL_BIN" ]] || fail 'fsutil.exe is unavailable; cannot enable case-sensitive NTFS directories.'
    windows_path="$(wslpath -w "$directory")"
    "$FSUTIL_BIN" file setCaseSensitiveInfo "$windows_path" enable >/dev/null 2>&1 || \
        fail "Could not enable NTFS case sensitivity on $directory."

    # Test nested names as Conda extracts them (share/terminfo/N and /n).
    probe="$directory/.case-sensitive-probe-$$"
    mkdir -p "$probe/share/terminfo/N" "$probe/share/terminfo/n" || \
        fail "Could not create the nested case-sensitivity probe under $directory."
    upper_inode="$(stat -c '%i' "$probe/share/terminfo/N")" || {
        rmdir "$probe/share/terminfo/N" "$probe/share/terminfo/n" 2>/dev/null || true
        rmdir "$probe/share/terminfo" "$probe/share" "$probe" 2>/dev/null || true
        fail "Could not inspect the uppercase probe directory under $directory."
    }
    lower_inode="$(stat -c '%i' "$probe/share/terminfo/n")" || {
        rmdir "$probe/share/terminfo/N" "$probe/share/terminfo/n" 2>/dev/null || true
        rmdir "$probe/share/terminfo" "$probe/share" "$probe" 2>/dev/null || true
        fail "Could not inspect the lowercase probe directory under $directory."
    }
    rmdir "$probe/share/terminfo/N" "$probe/share/terminfo/n" 2>/dev/null || true
    rmdir "$probe/share/terminfo" "$probe/share" "$probe" 2>/dev/null || true
    [[ "$upper_inode" != "$lower_inode" ]] || \
        fail "Case-sensitive lookup did not preserve distinct N/n directories under $directory."
    printf 'Case-sensitive NTFS verified: %s (N inode %s; n inode %s)\n' \
        "$directory" "$upper_inode" "$lower_inode"
}

write_cache_activation() {
    local hook_directory="$ENV_PREFIX/etc/conda/activate.d"
    local hook="$hook_directory/pgsr-cache-paths.sh"
    mkdir -p "$hook_directory"
    cat > "$hook" <<EOF
export CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS"
export CONDA_ENVS_PATH="$CONDA_ENVS_PATH"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export TMPDIR="$TMPDIR"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR"
export TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export CUDA_CACHE_PATH="$CUDA_CACHE_PATH"
export MPLCONFIGDIR="$MPLCONFIGDIR"
export TORCH_HOME="$TORCH_HOME"
export OPENCV_OPENCL_CACHE_DIR="$OPENCV_OPENCL_CACHE_DIR"
EOF
    chmod 0644 "$hook"
    printf 'Conda activation cache hook: %s\n' "$hook"
}

PROBE_DIR="$(mktemp -d "$CACHE_ROOT/preflight.XXXXXX")"
cleanup_probe() {
    rm -f "$PROBE_DIR/true" "$PROBE_DIR/link"
    rmdir "$PROBE_DIR"
}
trap cleanup_probe EXIT
cp /bin/true "$PROBE_DIR/true"
chmod +x "$PROBE_DIR/true"
"$PROBE_DIR/true" || fail 'The workspace filesystem cannot execute Linux binaries.'
ln -s true "$PROBE_DIR/link" || fail 'The workspace filesystem cannot create symlinks.'
[[ -L "$PROBE_DIR/link" ]] || fail 'The workspace symlink probe did not create a symlink.'
"$PROBE_DIR/link" || fail 'The workspace symlink cannot be followed.'
cleanup_probe
trap - EXIT

printf 'Workspace: %s\nEnvironment prefix: %s\n' "$ROOT" "$ENV_PREFIX"
printf 'Free before install: %s GiB\n' "$AVAILABLE_GIB"
printf 'Conda cache: %s\nPip cache: %s\nTemporary files: %s\n' \
    "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
printf 'Torch extensions: %s\nTriton cache: %s\nCUDA cache: %s\n' \
    "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
printf 'Install log: %s\n' "$LOG"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || \
    printf 'nvidia-smi unavailable; CUDA runtime smoke will be skipped.\n'
check_disk_budget 'before environment creation'

if [[ "${1:-}" == '--cache-hook-only' ]]; then
    [[ -x "$ENV_PREFIX/bin/python" ]] || fail "No usable Python at $ENV_PREFIX; cannot configure its activation hook."
    case_sensitive_preflight "$CONDA_PKGS_DIRS"
    case_sensitive_preflight "$ENV_PREFIX"
    write_cache_activation
    exit 0
fi

if [[ -e "$ENV_PREFIX" ]]; then
    fail "Environment path already exists; preserve it under a dated .failed name before retry: $ENV_PREFIX"
fi
if [[ -n "$(find "$CONDA_PKGS_DIRS" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    fail "Conda cache is not empty; preserve it under a dated .failed name before retry: $CONDA_PKGS_DIRS"
fi
mkdir -p "$ENV_PREFIX"
case_sensitive_preflight "$CONDA_PKGS_DIRS"
case_sensitive_preflight "$ENV_PREFIX"

"$CONDA_BIN" create --yes --prefix "$ENV_PREFIX" python=3.10 pip
write_cache_activation
check_disk_budget 'after environment creation'
PYTHON="$ENV_PREFIX/bin/python"
"$PYTHON" -m pip install \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.4.1+cu121' 'torchvision==0.19.1+cu121'
check_disk_budget 'after PyTorch installation'

# Install this exact precompiled wheel first. Constraints keep pgsr's declared
# unversioned gsplat dependency from selecting a different build or source.
"$PYTHON" -m pip install --no-deps "$GSPLAT_WHEEL"
"$PYTHON" -m pip install \
    --constraint "$CONSTRAINTS" \
    'pgsr[mesh]==1.0.0' \
    'opencv-python==4.10.0.84'
check_disk_budget 'after package installation'

"$PYTHON" -m pip check
"$PYTHON" -m pip freeze
printf 'Free after install: %s GiB\n' "$(df -BG --output=avail "$ROOT" | tail -n 1 | tr -cd '0-9')"
printf 'Setup complete. Log: %s\n' "$LOG"
