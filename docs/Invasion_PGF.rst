Invasion_PGF module
======================

Introduction
--------------------
**Invasion_PGF** is a Python package for applying Probability Generating Functions
[PGFs] to invasive processes, with an emphasis on the early spread of infectious disease.

The functions are based on the tutorial

`A tutorial on the use of probability generating functions in infectious disease modeling`_


Please cite the tutorial if using these algorithms.

		
Functions  (click on function name for full documentation)
-----------------------------------------------------------


Discrete time functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: Invasion_PGF
		   
.. autosummary::
   :toctree: functions

   R0
   extinction_prob
   active_infections
   completed_infections
   active_and_completed
   final_sizes

Continuous-time functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: Invasion_PGF
		   
.. autosummary::
   :toctree: functions
	     
   cts_time_R0
   cts_time_extinction_prob
   cts_time_active_infections
   cts_time_completed_infections
   cts_time_active_and_completed
   cts_time_final_sizes




.. _A tutorial on the use of probability generating functions in infectious disease modeling: Not_yet_published

