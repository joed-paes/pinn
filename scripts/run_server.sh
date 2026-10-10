# scripts/run_server.sh
#!/bin/bash
set -e
CONFIG=${1:-config/server.yaml}
EXP_NAME=${2:-mnist}
python src/train.py --config "$CONFIG" --name "$EXP_NAME" 2>&1 | tee -a experiments/_live.log