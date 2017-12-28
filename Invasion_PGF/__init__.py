r'''Functions for applying PGFs to infectious disease epidemics and 
potentially other invasive processes

This small package consists of a few algorithms for calculating quantities 
related to infectious disease transmission based on probability generating
functions.  

These functions require the input of a probability generating function for
the offspring distribution \mu(x).  All of these are generation-based.
The functions are:

- R0 : Estimates the reproductive based on approximating \mu'(1)

- extinction_prob : Calculates the probability of extinction after some
number of generations.  A large value gives a good approximation to the 
infinite generation limit.

- active_infections : Calculates the probability distribution for number of
actively infected individuals in a given generation.

- completed_infections : Calculates the probability distribution for number
of infections completed by the start of a given generation.

- active_and_completed : Calculates the joint distribution for number of 
active and completed infections at some generation.

- final_sizes : Calculates the probability distribution for size of small
outbreaks.
'''

__author__ = "Joel C. Miller"

__version__ = 0.1

import Invasion_PGF.Invasion_PGF
from Invasion_PGF.Invasion_PGF import *


