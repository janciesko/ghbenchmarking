#/bin/bash
BENCHMARK=$1
DATA_SIZE=8192 #131072 #1GB

DS=$DATA_SIZE
echo "gpu,ls,ts,veclen,elems,size,idx,idxsize,isAtomic,GUPS" | tee $BENCHMARK.LSTSVS.res
for LS in 1 2 4 8 16 32; do
    for TS in 1 2 4 8 16 32; do 
       for VS in 1; do 
          for reps in $(seq 1 2); do
	  ./$BENCHMARK 8192 $DS 7 0 $LS $TS 1| tee -a $BENCHMARK.res 
         done
       done	
    done
done



for LS in 1 2 4 8 16 32; do
    for TS in 1 2 4 8 16 32; do
       for VS in 1; do
       for reps in $(seq 1 2); do
           ./$BENCHMARK 8192 $DS 7 1 $LS $TS 1| tee -a $BENCHMARK.LSTSVS.res
         done
       done
    done
done