#!/bin/bash

session="$1"
if [ -z "$session" ]; then
  session="edl"
fi
tmux attach-session -t "$session" || tmux new-session -s "$session" \; send-keys '. /opt/img/effdl-venv/bin/activate' C-m
