#/bin/bash
BENCHMARK=$1
DATA_SIZE=1024 #64kB


export OMP_PROC_BIND=spread 
export OMP_PLACES=threads

export OMP_NUM_THREADS=72


DS=$DATA_SIZE
echo "gpu,ls,ts,veclen,elems,size,idx,idxsize,isAtomic,GUPS" | tee $BENCHMARK.LSTSVS.res 
for datasize in $(seq 1 20); do
    for reps in $(seq 1 3); do
	    ./$BENCHMARK 8192 $DS 7 0 32 32 1| tee -a $BENCHMARK.LSTSVS.res 
    done	
  let DS=$DS*2
done

echo "gpu,ls,ts,veclen,elems,size,idx,idxsize,isAtomic,GUPS" | tee $BENCHMARK.LSTSVS_atomic.res 
let DS=$DATA_SIZE
for datasize in $(seq 1 20); do
    for reps in $(seq 1 3); do
        ./$BENCHMARK 8192 $DS 7 1 32 32 1| tee -a $BENCHMARK.LSTSVS_atomic.res 
    done
  let DS=$DS*2
done
