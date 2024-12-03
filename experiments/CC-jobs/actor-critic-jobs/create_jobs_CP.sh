#!/bin/bash

INIT_VALUES="-5 0 5"

BASE_SCRIPT="job_CP.sh"

for INIT in $INIT_VALUES; do
    NEW_SCRIPT="CP_init_$INIT.sh"
    
    sed "s/INIT_VALUE=0/INIT_VALUE=$INIT/" $BASE_SCRIPT > $NEW_SCRIPT

    chmod +x $NEW_SCRIPT
    
    echo "Generated script: $NEW_SCRIPT"
done

echo "All scripts created!"

echo "--------------------------------------------------------------"