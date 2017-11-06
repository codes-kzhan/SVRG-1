#!/bin/bash

cgcreate -g memory:/MemGroup # in case we do not have such a user

RED='\033[0;31m'
NC='\033[0m' # No Color

for i in 32 16 8
do
    echo "${RED}Testing criteo under ${i} G available MEM...${NC}\n\n"
    MEM_SIZE=$(($i * 1024 * 1024 * 1024))
    echo $MEM_SIZE > /sys/fs/cgroup/memory/MemGroup/memory.limit_in_bytes
    #/usr/local/bin/matlab -nodisplay -nodesktop -r "svm_main();exit;"
    cgexec -g memory:MemGroup /usr/local/bin/matlab -nodisplay -nodesktop -r "warmup; criteo_main('${i}G'); exit;"
done
