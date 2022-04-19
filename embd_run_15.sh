echo -e "\nseqlen = 30, word2vec (k = 5, stride = 2)\n"
python3 ./main.py --target 1 --model 0 --embd 2 --seqlen 30 --kmer 5 --stride 2
python3 ./main.py --target 5 --model 0 --embd 2 --seqlen 30 --kmer 5 --stride 2
python3 ./main.py --target 6 --model 0 --embd 2 --seqlen 30 --kmer 5 --stride 2
python3 ./main.py --target 7 --model 0 --embd 2 --seqlen 30 --kmer 5 --stride 2
#python3 ./main.py --target 8 --model 0 --embd 2 --seqlen 30 --kmer 5 --stride 2
 
