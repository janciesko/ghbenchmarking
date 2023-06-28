#/bin/bash
BENCHMARK=$1
DATA_SIZE=100000000

export OMP_PROC_BIND=spread 
export OMP_PLACES=threads

export OMP_NUM_THREADS=72

DS=$DATA_SIZE
echo "ls,ts,vs,elems,size,size_total,setTime,copyTime,scaleTime,addTime,triadTime" | tee $BENCHMARK_LSTSVS.res 
for LS in 256 512 1024; do
    for TS in 128 256 512 1024 ; do 
       for VS in 1; do 
          for reps in $(seq 1 2); do
	          ./$BENCHMARK $DS 3 $LS $TS $VS| tee -a $BENCHMARK_LSTSVS.res 
         done
       done	
    done
done

