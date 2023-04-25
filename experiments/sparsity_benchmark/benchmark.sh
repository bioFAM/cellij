
#!/bin/bash

# Create data
python ../../cellij/benchmark/generate.py -n 200 -d 400 -d 400 -d 400 -ll normal -ll normal -ll normal -k 10 0 0 -fsd uniform --seed 0 --out-dir synthetic

# Train models
lrs=(0.003)
n_factors=(10)
sparsity_priors=(Horseshoe Spikeandslab-Beta Spikeandslab-ContinuousBernoulli Spikeandslab-RelaxedBernoulli Lasso)

for lr in ${lrs[@]}; do
    for factors in ${n_factors[@]}; do
        for sparsity in ${sparsity_priors[@]}; do
            echo "factors: $factors"
            echo "sparsity: $sparsity"
            python ../../cellij/benchmark/train.py --seed 0 --model mofa -dd synthetic -k $factors -sp $sparsity -ll normal -ll normal -ll normal --epochs 100 -lr $lr -ve 100 --out-dir training_output_$sparsity
            echo "done"
            echo ""
        done
    done
done