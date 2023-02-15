# Scripts to produce data files from CMU dataset

1. Download .amc files from CMU mocap dataset e.g walking http://mocap.cs.cmu.edu/search.php?subjectnumber=7
2. Use `amc_interpolation.py` to interpolate the .amc from 120hz to mujoco 66.67hz
3. Use `load.py` to create data files for CoIL

We have included the preprocessed data for the humanoid and cheetah tasks in a separate "data" folder.