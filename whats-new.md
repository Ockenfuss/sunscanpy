# What's New
## 0.1.3
- The sky signal is composed of sky emission and receiver noise. In reality, the latter dominates. This is now taken into account in the signal simulation.
However, changes in the results are minor (~ 1e-4 deg) and it is generally NOT necessary to repeat simulations performed with older SunscanPy versions.
The public API of `SignalSimulationEstimator` remains unchanged and continues to use the `sky_signal` keyword parameter. Internally, this value is now interpreted as receiver noise.

## 0.1.2
- Bugfix: sun position calculation was wrong on southern hemisphere (sign flip not handled correctly)
- Solar positions are returned as numpy arrays.
- Add missing matplotlib dependency
- Remove the suggestion for a scan pattern in the first tutorial. This will be soon covered in detail in the corresponding publication.

## 0.1.1
- Rename `elevation` to `altitude` in `SunObject`.
- Reduce log verbosity of sun object

## 0.1.0
Initial release of SunscanPy to PyPI.