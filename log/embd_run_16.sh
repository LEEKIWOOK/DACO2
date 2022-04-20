echo -e "\nseqlen = 30, word2vec (k = 7, stride = 2)\n"
python3 ./main.py --target 1 --model 0 --embd 2 --seqinfo full --kmer 7 --stride 2
python3 ./main.py --target 7 --model 0 --embd 2 --seqinfo full --kmer 7 --stride 2
