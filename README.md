# Analysis of Actuator Selection for Discrete-time linear systems with Multiplicative Noise

This is a experimental simulation study of actuator selection for discrete-time linear systems with multiplicative noise. The analysis starts on a simple system with closed loop actuator control with state and input dependent multiplicative noise. The goal is to expand analysis to dynamic games with both control and adversarial actuators and finally to the case of both additive and multiplicative noise.

#### Test-Branch4 Targets:

##### Working
- Simulation of actuator selection

##### Done
- Rework cost function algorithm
  - cost matrix fails to converge within given time
  - feedback matrix for case when cost fails to converge
- Rework actuator selection
  - selection when no choices are feasible/converge in costs
- Visualizing actuator selection
  - plotting feasible vs infeasible selections and costs
