#!/usr/bin/env bash
set -euo pipefail
exec docker exec -i assignment3-k3s kubectl "$@"
