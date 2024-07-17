#!/bin/bash

usage() {
    echo "Usage: $0 <job_id> [--last_checkpoint | --tensorboard]"
    echo "  <job_id>           The job ID (required)"
    echo "  --last_checkpoint  Perform last checkpoint action"
    echo "  --tensorboard      Perform tensorboard action"
    exit 1
}

# Check if at least two arguments are provided
if [ $# -lt 2 ]; then
    usage
fi

# Assign the first argument to job_id
job_id=$1
shift

# Parse command-line options
case "$1" in
    --last_checkpoint)
        step_number=$(ls /scratch-shared/fomo_logs/$job_id/checkpoints | sort -V | tail -1)
        echo "Copying ${step_number} of job ID: $job_id"

        mkdir -p backup/$job_id/checkpoints
        cp -r /scratch-shared/fomo_logs/$job_id/checkpoints/$step_number backup/$job_id/checkpoints
        ;;
    --tensorboard)
        echo "Copying tensorboard logs for job ID: $job_id"
        
        mkdir -p backup/$job_id/
        cp -r /scratch-shared/fomo_logs/$job_id/tensorboard backup/$job_id/
        ;;
    *)
        echo "Error: Invalid option. Must be either --last_checkpoint or --tensorboard."
        usage
        ;;
esac
