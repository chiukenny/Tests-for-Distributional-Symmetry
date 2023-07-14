## Requires 'events_anomalydetection_v2.features.h5' saved in './data/'
## The dataset can be downloaded from the following link:
## https://zenodo.org/record/6466204


# Seed for reproducibility
Random.seed!(1)

# Read data
fid = h5open(dir_dat*"events_anomalydetection_v2.features.h5", "r")
LHC = read(fid["df/block0_values"])
n_LHC = size(LHC)[2]

# Randomly split the data into a training and test set
n_tr = floor(Int64, n_LHC/2)
inds = sample(1:n_LHC, n_tr, replace=false)
LHC_tr = LHC[[1,2,8,9], inds]

# Save the data
fid2 = h5open(dir_dat*"LHC.h5", "w")
write(fid2, "train", LHC_tr)
write(fid2, "data", LHC[[1,2,8,9],Not(inds)])

close(fid)
close(fid2)