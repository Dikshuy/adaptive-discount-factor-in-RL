#!/bin/bash

Q_INIT_VALUES="0 1 5 10"
ALPHA_VALUES="0.1 0.5 1"

BASE_SCRIPT="job_simplegrid.sh"

JOB_DIR="./job_scripts"
mkdir -p $JOB_DIR

for INIT in $Q_INIT_VALUES; do
    for ALPHA in $ALPHA_VALUES; do
        JOB_FILE="$JOB_DIR/job_init_${INIT}_alpha_${ALPHA}.sh"
        
        sed -e "s/Q_INIT=0/Q_INIT=$INIT/" \
            -e "s/ALPHA=0.1/ALPHA=$ALPHA/" \
            $BASE_SCRIPT > $JOB_FILE

        chmod +x ${JOB_FILE}

        echo "Generate job file: $JOB_FILE"

        # sbatch $JOB_FILE
        bash $JOB_FILE

        echo "submitted job: $JOB_FILE"

        echo "================================="
  done
done

echo "Job scripts created in ${JOB_DIR}"

echo "All scripts created and submitted!"
echo "--------------------------------------------------------------"