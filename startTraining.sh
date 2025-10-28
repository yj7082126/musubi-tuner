#!/bin/bash

CLUSTER=`sacctmgr show cluster  -P | tail -n 1 | cut -f1 -d "|"`
EXIT_STATUS=0
echo -e "Current Directory: `pwd`"

# Start the job, including launching JN, then collect info about the job
echo -e "\nSpinning up your Training! (Give me about 5s...)"
JOBID="$(sbatch --parsable submit2.slurm)"
sleep 15
JOB_NODELIST="$(squeue -j $JOBID -o %N | grep ^e)"
BATCH_NODE=`scontrol show job $JOBID | grep BatchHost | cut -d"=" -f2`
# Report information about the job
echo -e "\nJobID: $JOBID"
echo -e "NodeList:   $JOB_NODELIST"
echo -e "BatchNode:  $BATCH_NODE"

echo -e

exit $EXIT_STATUS