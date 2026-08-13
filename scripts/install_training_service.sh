#!/usr/bin/env bash
set -euo pipefail

install -m 0644 /data/code/hf_space/barbados-dapt.service \
    /etc/systemd/system/barbados-dapt.service
systemctl daemon-reload
systemctl enable --now barbados-dapt.service
