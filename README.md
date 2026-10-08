# RTRL-Flight-controller
A flight controller for a Cessna 172 in JSBSim that predicts aileron, elevator and rudder deflections to hold a target attitude. It uses Real-Time Recurrent Learning (RTRL) to adapt its weights in flight, aiming to recover from actuator faults. It is benchmarked against PID and BPTT-LSTM baselines under nominal, fault, wind and combined scenarios.
