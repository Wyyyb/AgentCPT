#/bin/bash

set -xv

TASK=cpu_tmux
VERSION=0.1

# cd ./math_serving/reasonreason && git checkout main && git pull && cd -

sudo docker build --network host -t harbor.xaminim.com/minimax-dialogue/$TASK:${VERSION} . -f docker/Dockerfile_tmux_cpu_yb

sudo docker push harbor.xaminim.com/minimax-dialogue/$TASK:${VERSION}
