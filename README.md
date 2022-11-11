# Actuator Selection for Dynamical Networks with Multiplicative Noise

Submitted for IFAC 2023

### Abstract
We propose a greedy algorithm for actuator selection considering multiplicative noise in the dynamics and actuator architecture of a discrete-time, linear system network model. We show that the resultant architecture achieves mean-square stability with lower control costs and for smaller actuator sets than the deterministic model, even in the case of modeling uncertainties. Networks with multiplicative noise may fail to be mean-square stabilized for small actuator sets, leading to a failure of the greedy algorithm. To account for this, we propose a multi-metric greedy algorithm that allows actuator sets to be evaluated effectively even when none of them stabilize the system. We illustrate our results on networks with multiplicative noise in the open-loop dynamics and the actuator inputs, and we analyze control costs for random graphs of different network sizes and generation parameters.

### Test Files
- [Model Test File](ActuatorSelection_Test3_2.ipynb)
    - Test models generated in [ModelFiles](ModelFiles.ipynb)
      - Model 10 - test for multiplicative noise in open-loop dynamics
      - Model 11 - test for multiplicative noise in actuator set
- [Statistical Test File](ActuatorSelection_Test4_2.ipynb)
    - Generate and test realizations of random graphs with data stored to pickle file in [system_test](system_test)
- [Statistical Test Data Visualization](ActuatorSelection_Test4_3.ipynb)
    - Visualizer for test data from Statistical Test File
- [ModelFiles](ModelFiles.ipynb)
    - Generate test models for [Model Test File](ActuatorSelection_Test3_2.ipynb)
- Function Files
    - [System Definition](functionfile_system_definition.py)
    - [Cost Calculation and Actuator Selection](functionfile_system_mplcost.py)
