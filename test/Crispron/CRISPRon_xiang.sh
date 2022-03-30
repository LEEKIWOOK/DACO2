#!/bin/bash
SDIR='/home/kwlee/Projects_gflas/DACO2/tools/crispron'
BINDIR='/home/kwlee/Projects_gflas/DACO2/tools/crispron/bin'
DATADIR='/home/kwlee/Projects_gflas/DACO2/tools/crispron/data'

INDIR='/home/kwlee/Projects_gflas/DACO2/data/input/Raw_data/Xiang'
OUTDIR='/home/kwlee/Projects_gflas/DACO2/output/Crispron/Xiang'

if [[ -z $OUTDIR ]]; then
	echo "Needs two arguments to run e.g. $0 test/seq.fa test/outdir" 1>&2
	exit 1
fi

mkdir -p $OUTDIR || exit 1
which python3 || exit 1
RNAfold --version || exit 1

if [[ ! -s $INDIR/30mers.fa ]]; then
	python3 ./readfile.py --data Xiang
fi

echo "#Running CRISPROff pipeline"
$BINDIR/CRISPRspec_CRISPRoff_pipeline.py \
	--guides $INDIR/23mers.fa \
	--specificity_report $OUTDIR/CRISPRspec.tsv \
	--guide_params_out $OUTDIR/CRISPRparams.tsv \
	--duplex_energy_params $DATADIR/model/energy_dics.pkl \
	--no_azimuth || exit 1

echo "#Running CRISPRon evaluation"
$BINDIR/DeepCRISPRon_eval.py $OUTDIR $INDIR/30mers.fa $OUTDIR/CRISPRparams.tsv $DATADIR/deep_models/best/*/  2>&1 | grep -v tracing

echo "#All done" 1>&2