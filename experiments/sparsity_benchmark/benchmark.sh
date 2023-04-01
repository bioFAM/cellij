
#!/bin/bash

# Create data
python ../../cellij/benchmark/generate.py -n 200 -d 400 -d 400 -d 400 -ll normal -ll normal -ll normal -k 10 0 0 -fsd uniform --seed 0 --out-dir synthetic

# Train models
lrs=(0.001 0.01 0.1)
n_factors=(5 10 15 20 25)
sparsity_priors=(Horseshoe, Spikeandslab-Beta, Spikeandslab-ContinuousBernoulli)

for lr in ${lrs[@]}; do
    for factors in ${n_factors[@]}; do
        for sparsity in "${sparsity_priors[@]}"; do
            echo "factors: $factors"
            echo "sparsity: $sparsity"
            python ../../cellij/benchmark/train.py --seed 0 --model mofa -dd synthetic -k $factors -sp $sparsity_priors -ll normal -ll normal -ll normal --epochs 10000 -lr $lr -ve 100 --out-dir training_output
            echo "done"
            echo ""
        done
    done
done