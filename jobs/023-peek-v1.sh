# peek at the v1 training log
grep -E "params|step|EVAL|Error|error" /workspace/logs/021-train.log | tail -15
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
