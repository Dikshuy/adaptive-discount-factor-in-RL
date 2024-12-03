#!/bin/bash

INIT_VALUES="-5 0 5"
GAMMA_VALUES="0.1 0.25 0.5 0.75 0.95 0.99"

BASE_SCRIPT="job_P.sh"

SCRIPT_DIR="./generated_scripts"
mkdir -p $SCRIPT_DIR

for INIT in $INIT_VALUES; do
    for GAMMA in $GAMMA_VALUES; do
        NEW_SCRIPT="$SCRIPT_DIR/P_init_${INIT}_gamma_${GAMMA}.sh"

        sed -e "s/INIT_VALUE=0/INIT_VALUE=$INIT/" \
            -e "s/GAMMA_VALUE=0.1/GAMMA_VALUE=$GAMMA/" \
            $BASE_SCRIPT > $NEW_SCRIPT

        chmod +x $NEW_SCRIPT

        echo "Generated script: $NEW_SCRIPT"

        sbatch $NEW_SCRIPT
        echo "Submitted job: $NEW_SCRIPT"
    done
done

echo "All scripts created and submitted!"
echo "--------------------------------------------------------------"
