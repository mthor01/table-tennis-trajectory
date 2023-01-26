There is a required run order for everything to work which goes as follows:

detect_cuts.py
pose_estimation.py
simulation_data_gen.py
3d_projection_evo.py
evaluation.py

Please note that simulation data gen only constructs one simple trajectory right now, as it is easier to work with in
further scripts. 3d_projection_evo.py is very slow as of now so I set the population size and the number of generations
very low. Therefore there is very limited data with bad estimations, resulting in a very uninteresting outcome
of evaluation.py.
