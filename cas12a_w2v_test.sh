#!/bin/sh
for j in full i3 i2 i1 spacer d1 d2 d3 d4 d5; do
	echo -e "Log: Cas12a/onehot/seq_${j}\n"
	echo -e "python3 ./main.py --model 0 --target 8 --embd 0 --seqinfo ${j}\n"
	time python3 ./main.py --model 0 --target 8 --embd 0 --seqinfo ${j}

	echo -e "Log: Cas12a/embdtb/seq_${j}\n"
	echo -e "python3 ./main.py --model 0 --target 8 --embd 1 --seqinfo ${j}\n"
	time python3 ./main.py --model 0 --target 8 --embd 1 --seqinfo ${j}

	#for k in `seq 3 8`; do #kmer
	#	for s in 1 2; do #stride
			k=5
			s=2
			echo -e "Log: Cas12a/word2vec_k${k}_s${s}/seq_${j}\n"
			echo -e "python3 ./main.py --model 0 --target 8 --embd 2 --seqinfo ${j} --kmer ${k} --stride ${s}\n"
			time python3 ./main.py --model 0 --target 8 --embd 2 --seqinfo ${j} --kmer ${k} --stride ${s}
	#	done
	#done
done
