#/bin/bash
BENCHMARK=$1
DATA_SIZE=33554432


export OMP_PROC_BIND=spread 
export OMP_PLACES=threads

export OMP_NUM_THREADS=72

DS=$DATA_SIZE
echo "elems,size,size_total,setTime,copyTime,scaleTime,addTime,triadTime" | tee $BENCHMARK_LSTSVS.res 
for LS in 1 2 4 8 16 32 64 128; do
    for TS in 1 2 4 8 16 32 64; do 
       for VS in 1; do 
          for reps in $(seq 1 2); do
	          ./$BENCHMARK $DS 4 $LS $TS $VS| tee -a $BENCHMARK_LSTSVS.res 
         done
       done	
    done
done

