# A bunch of OpenAI Gym environments

These envs return the feet positions and velocities in the info dict under
`feet_pos` and `feet_velp`, as well as the same positions relative to the torso
(`rel_feet_pos` and `rel_feet_velp`). All envs are parametric (can have their
masses and lengths changed). In HalfCheetah, due to the way the model is
parametrized, the parametric HalfCheetah is not identical to the original one
(but is very similar). Instead, the `GaitTrackHalfCheetahOriginal-v0` can be
used if an identical model is needed (but it cannot have its link lengths
changed).

## Envs

  1. `GaitTrackAnt-v0`
  1. `GaitTrackWalker2d-v0`
  1. `GaitTrackHopper-v0`
  1. `GaitTrackHalfCheetah-v0`
  1. `GaitTrackHalfCheetahOriginal-v0`

