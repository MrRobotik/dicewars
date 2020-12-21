#!/bin/bash
cd ../../../../
. path.sh
for i in {1..100}; do
	python3 ./scripts/dicewars-tournament.py -r -g 4 -n 1 --ai-under-test xkucer95
	cp dicewars/ai/xkucer95/models/policy_model.pt dicewars/ai/xkucer95/models/policy_model-$i.pt
	python3 ./dicewars/ai/xkucer95/policy_training.py dicewars/ai/xkucer95/models/policy_model_trn.dat >> dicewars/ai/xkucer95/models/loss-$i.csv
	# Eval...
	mv dicewars/ai/xkucer95/ai.py dicewars/ai/xkucer95/models/temp
	mv dicewars/ai/xkucer95/models/ai.py dicewars/ai/xkucer95/
	python3 ./scripts/dicewars-tournament.py -r -g 4 -n 1 --ai-under-test xkucer95 >> dicewars/ai/xkucer95/models/eval-$i.txt
	mv dicewars/ai/xkucer95/ai.py dicewars/ai/xkucer95/models/
	mv dicewars/ai/xkucer95/models/temp/ai.py dicewars/ai/xkucer95/
done
