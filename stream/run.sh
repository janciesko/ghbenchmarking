#/bin/bash
BENCHMARK=$1
DATA_SIZE=1024 #64kB


export OMP_PROC_BIND=spread 
export OMP_PLACES=threads

export OMP_NUM_THREADS=72


DS=$DATA_SIZE
echo "elems,size,size_total,setTime,copyTime,scaleTime,addTime,triadTime" | tee $BENCHMARK.res 
for datasize in $(seq 1 20); do
    for reps in $(seq 1 3); do
	    ./$BENCHMARK 8192 $DS 7 0 32 32 1| tee -a $BENCHMARK.LSTSVS.res 
    done	
  let DS=$DS*2
done
