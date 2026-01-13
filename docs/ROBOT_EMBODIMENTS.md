# Robot Embodiment Configurations

This document captures the required assets and kinematic parameters for robot embodiments
added via `policy_configs/robot_embodiments.json`. The task generator consumes these entries
so it can build consistent action/observation spaces for new robots.

## Newly Added Robots

| Robot ID | URDF Asset | USD Asset | DOFs | Gripper DOFs | EE Frame | Default Joint Positions | Reach Radius (m) | Base Height (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ur5e` | `robots/ur5e/ur5e.urdf` | `robots/ur5e/ur5e.usd` | 6 | 2 | `tool0` | `[0.0, -1.571, 1.571, -1.571, -1.571, 0.0]` | 0.85 | 0.0 |
| `ur10e` | `robots/ur10e/ur10e.urdf` | `robots/ur10e/ur10e.usd` | 6 | 2 | `tool0` | `[0.0, -1.571, 1.571, -1.571, -1.571, 0.0]` | 1.30 | 0.0 |
| `kuka_iiwa` | `robots/kuka_iiwa/iiwa14.urdf` | `robots/kuka_iiwa/iiwa14.usd` | 7 | 2 | `iiwa_link_ee` | `[0.0, 0.6, 0.0, -1.7, 0.0, 1.0, 0.0]` | 0.80 | 0.0 |
| `kinova_gen3` | `robots/kinova_gen3/gen3.urdf` | `robots/kinova_gen3/gen3.usd` | 7 | 2 | `tool_frame` | `[0.0, 0.5, 0.0, 1.5, 0.0, 1.2, 0.0]` | 0.90 | 0.0 |
| `tiago` | `robots/tiago/tiago.urdf` | `robots/tiago/tiago.usd` | 7 | 2 | `gripper_link` | `[0.2, -1.34, -0.2, 1.94, -1.57, 1.37, 0.0]` | 0.90 | 0.35 |

## Notes

- Asset paths are relative to the robot asset root expected by the pipeline.
- Kinematic values mirror the parameters in `policy_configs/robot_embodiments.json`.
- Update this document whenever new robot IDs are added to keep task generation and
  asset requirements aligned.
