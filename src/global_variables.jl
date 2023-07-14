## Constant variables for convenience


# Default test names
const GV_2S = "2SMMD"
const GV_MMD = "MMD"
const GV_NMMD = "NMMD"
const GV_CW = "CW"
const GV_KCI = "KCI"
const GV_CP = "CP"


# North geomagnetic pole coordinates
NP_lat = 80.7 * π / 180
NP_lon = -72.7 * π / 180
const GV_NPx = cos(NP_lat) * cos(NP_lon)
const GV_NPy = cos(NP_lat) * sin(NP_lon)
const GV_NPz = sin(NP_lat)