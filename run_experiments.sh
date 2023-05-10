for experiment in range-mean-0 range-mean-100 range-min-0 range-min-100 zscore-mean-0 zscore-mean-100 zscore-min-0 zscore-min-100
do
    echo "Running experiment: $experiment"
    python train.py +experiment=11-05-2023/normalization/"$experiment"
done

echo "Done!"
