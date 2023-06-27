#/bin/bash
BENCHMARK=$1
DATA_SIZE=8192  #131072 #1GB

#8192 64KB
#131072 #1GB

export OMP_PROC_BIND=spread 
export OMP_PLACES=threads

export OMP_NUM_THREADS=72

DS=$DATA_SIZE
echo "gpu,ls,ts,veclen,elems,size,idx,idxsize,isAtomic,GUPS" | tee $BENCHMARK.LSTSVS.res
for LS in 1 2 4 8 16 32 64 128; do
    for TS in 1 2 4 8 16 32 64 128; do 
       for VS in 1; do 
          for reps in $(seq 1 2); do
	          ./$BENCHMARK 8192 $DS 7 0 $LS $TS 1| tee -a $BENCHMARK.res 
         done
       done	
    done
done



for LS in 1 2 4 8 16 32 64 128; do
    for TS in 1 2 4 8 16 32 64 128; do
       for VS in 1; do
       for reps in $(seq 1 2); do
           ./$BENCHMARK 8192 $DS 7 1 $LS $TS 1| tee -a $BENCHMARK.LSTSVS_atomic.res
         done
       done
    done
done
