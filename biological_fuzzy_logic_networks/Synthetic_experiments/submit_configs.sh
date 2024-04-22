#!/bin/bash
source /opt/share/anaconda3-2019.03/x86_64/etc/profile.d/conda.sh
conda activate biofuzznet2
echo $CONDA_DEFAULT_ENV

# config_folder=/dccstor/ipc1/CAR/BFN/Model/Noise/Configs/

# for file in $config_folder*.json
#     do 
#         # echo ${file}
#         jbsub -q x86_6h -cores 1+1 -mem 50g python /u/adr/Code/biological_fuzzy_logic_networks/biological_fuzzy_logic_networks/Synthetic_experiments/training_noise.py $file
#     done


# for n_gates in {0..22}
#     do 
#         # echo ${n_gates}
#         jbsub -q x86_6h -cores 1+1 -mem 50g python /u/adr/Code/biological_fuzzy_logic_networks/biological_fuzzy_logic_networks/Synthetic_experiments/learn_gates.py /u/adr/Code/biological_fuzzy_logic_networks/biological_fuzzy_logic_networks/Synthetic_experiments/base_gate_config.json --n_gates $n_gates
#     done

datadir=/dccstor/ipc1/CAR/BFN/Data/CNoRFuzzy/
outdir=/dccstor/ipc1/CAR/BFN/Model/CNoRFuzzy/
jbsub -q x86_24h -mem 50g python /u/adr/Code/biological_fuzzy_logic_networks/biological_fuzzy_logic_networks/Synthetic_experiments/CNoRFuzzy_data_CV.py --model ${datadir}optimised_structure_DREAM_reduced.sif --data ${datadir}simulated_data_DREAM_reduced.csv --foldnum 100 --type fuzz --outputfolder $outdir --epochs 700  --learningrate 0.002 --batchsize 3 --rounds 1