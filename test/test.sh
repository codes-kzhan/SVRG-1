#!/bin/bash

cgcreate -g memory:/MemGroup # in case we do not have such a user

for i in {8..2..-2}
do
    MEM_SIZE=$(($i * 1024 * 1024 * 1024))
    echo $MEM_SIZE > /sys/fs/cgroup/memory/MemGroup/memory.limit_in_bytes
    #/usr/local/bin/matlab -nodisplay -nodesktop -r "svm_main();exit;"
    cgexec -g memory:MemGroup /usr/local/bin/matlab -nodisplay -nodesktop -r "warmup; avazu_main('${i}G'); exit;"
done
