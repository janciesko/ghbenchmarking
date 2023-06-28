#/bin/bash
BENCHMARK=$1
DATA_SIZE=33554432

#$33554432 //256MB

#131072 #1MB

#8192 64KB
#131072 #1MB


export OMP_PROC_BIND=spread 
export OMP_PLACES=threads

export OMP_NUM_THREADS=72

DS=$DATA_SIZE
echo "gpu,ls,ts,veclen,elems,size,idx,idxsize,isAtomic,GUPS" | tee $BENCHMARK.LSTSVS.res 
for LS in 8; do
    for TS in 1 2 4 8 16 32 64; do 
       for VS in 1 2 4 8 16; do 
          for reps in $(seq 1 2); do
	          ./$BENCHMARK 8192 $DS 7 0 $LS $TS $VS| tee -a $BENCHMARK.LSTSVS.res 
         done
       done	
    done
done

#echo "gpu,ls,ts,veclen,elems,size,idx,idxsize,isAtomic,GUPS" | tee $BENCHMARK.LSTSVS_atomic.res 

#for LS in 1 2 4 8 16 32 64 128; do
#    for TS in 1 2 4 8 16 32 64; do
#       for VS in 1; do
#       for reps in $(seq 1 2); do
#           ./$BENCHMARK 8192 $DS 7 1 $LS $TS $VS| tee -a $BENCHMARK.LSTSVS_atomic.res
#         done
#       done
#    done
#done
