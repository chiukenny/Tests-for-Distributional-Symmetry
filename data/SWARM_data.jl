## Requires 'SWARM_DATA.csv' saved in './data/'
## The dataset can be downloaded from the following link:
## https://github.com/lchristie/estimating_symmetries/tree/main/Geomag_data


# Seed for reproducibility
Random.seed!(1)

# Read data
raw_SWARM = CSV.read(dir_dat*"SWARM_DATA.csv", DataFrame, select=[:LAT,:LONG,:RAD,:F_NT])
n_SWARM = size(raw_SWARM)[1]

# Transform the lat-lon data to Cartesian
lat = raw_SWARM[:, :LAT]
lon = raw_SWARM[:, :LONG]
rad = raw_SWARM[:, :RAD]
rad = rad ./ maximum(rad)
SWARM = zeros(4, n_SWARM)
SWARM[1,:] = rad .* cos.(lat.*(π/180)) .* cos.(lon.*(π/180))
SWARM[2,:] = rad .* cos.(lat.*(π/180)) .* sin.(lon.*(π/180))
SWARM[3,:] = rad .* sin.(lat.*(π/180))

# Standardize the magnetic field strength
SWARM[4,:] = standardize( reshape(raw_SWARM[:,:F_NT],1,n_SWARM) )

# Randomly split the data into a training and test set
n_tr = floor(Int64, n_SWARM/2)
inds_tr = sample(1:n_SWARM, n_tr, replace=false)
SWARM_tr = SWARM[:, inds_tr]

# Save the data
fid = h5open(dir_dat*"SWARM.h5", "w")
write(fid, "train", SWARM_tr)
write(fid, "data", SWARM[:,Not(inds_tr)])
close(fid)