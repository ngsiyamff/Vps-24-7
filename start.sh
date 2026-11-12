#!/bin/bash
ttyd -W -p ${PORT:-10000} -c admin:12345 tmux new -A -s cloud
