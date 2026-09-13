# Main Results

Generated from `data/results/sweep_results.parquet` via `rtrl_flight.analysis.aggregate.write_markdown_table` -- do not hand-edit.

| Scenario | Controller | Attitude RMSE (rad) | Episode Return | Control Jerk | Time to Recover (steps) |
|---|---|---|---|---|---|
| nominal | bptt_lstm | 0.3192 +/- 0.1122 | -33.2 +/- 22.9 | 0.0017 | N/A |
| nominal | pid | 0.1749 +/- 0.0404 | -9.7 +/- 4.4 | 0.0038 | N/A |
| nominal | rtrl_rtu | 1.5822 +/- 0.1732 | -1356.8 +/- 183.1 | 0.0788 | N/A |
| fault | bptt_lstm | 0.5446 +/- 0.0910 | -90.6 +/- 28.5 | 0.0017 | 0.0 |
| fault | pid | 0.2216 +/- 0.0221 | -15.0 +/- 3.0 | 0.0044 | N/A |
| fault | rtrl_rtu | 1.5340 +/- 0.2275 | -1070.1 +/- 598.9 | 0.0684 | N/A |
| wind | bptt_lstm | 0.3320 +/- 0.1010 | -35.9 +/- 20.6 | 0.0031 | N/A |
| wind | pid | 0.2047 +/- 0.0282 | -13.8 +/- 3.8 | 0.0323 | N/A |
| wind | rtrl_rtu | 1.5980 +/- 0.0831 | -1073.2 +/- 284.2 | 0.0721 | N/A |
| combined | bptt_lstm | 0.4702 +/- 0.0823 | -68.2 +/- 22.1 | 0.0027 | N/A |
| combined | pid | 0.2531 +/- 0.0306 | -20.4 +/- 4.8 | 0.0317 | 73.0 |
| combined | rtrl_rtu | 1.6927 +/- 0.0647 | -1185.8 +/- 236.2 | 0.0688 | N/A |
| ablation | rtrl_rtu | 2.0484 +/- 0.2618 | -3719.7 +/- 3821.1 | 0.1021 | N/A |
