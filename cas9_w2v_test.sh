#!/bin/sh
for i in `seq 1 7`; do #target data
j=full
#	for j in full i3 i2 i1 spacer d1 d2 d3 d4 d5; do
		echo -e "Log: Cas9/target${i}/onehot/seq_${j}\n"
		echo -e "python3 ./main.py --model 0 --target ${i} --embd 0 --seqinfo ${j}\n"
		time python3 ./main.py --model 0 --target ${i} --embd 0 --seqinfo ${j}

		echo -e "Log: Cas9/target${i}/embdtb/seq_${j}\n"
		echo -e "python3 ./main.py --model 0 --target ${i} --embd 1 --seqinfo ${j}\n"
		time python3 ./main.py --model 0 --target ${i} --embd 1 --seqinfo ${j}

		for k in `seq 3 8`; do #kmer
			for s in 1 2; do #stride
				echo -e "Log: Cas9/target${i}/word2vec_k${k}_s${s}/seq_${j}\n"
				echo -e "python3 ./main.py --model 0 --target ${i} --embd 2 --seqinfo ${j} --kmer ${k} --stride ${s}\n"
				time python3 ./main.py --model 0 --target ${i} --embd 2 --seqinfo ${j} --kmer ${k} --stride ${s}
			done
		done
#	done
done
