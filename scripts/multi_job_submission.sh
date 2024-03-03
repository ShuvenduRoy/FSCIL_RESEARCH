#!/bin/bash
cd scripts/bash || exit

for i in $(seq "$1" "$2"); do
	echo "Submitting job id:  ${i} "
	sbatch "exp${i}.sh"
done
