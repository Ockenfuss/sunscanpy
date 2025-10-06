
<div align="center">
  <img src="assets/icons/SunscanPyLogo.svg" alt="SunscanPy Logo" width="200" height="auto">
  <!-- <h1>SunscanPy</h1> -->
  <p><em>A python library to calibrate the pointing of your weather or cloud radar based on the position of the Sun!</em></p>
</div>

## Description

Precise knowledge of the pointing direction is essential for weather and cloud radars. Radars with scanning capability can be calibrated using the sun as a microwave emitting target with a precisely defined position in the sky.
SunscanPy provides tools to evaluate radar measurements of the sun, derive the mispointing of your antenna and actively correct misalignments using the scanner motors.

For more information, see the two tutorial jupyter notebooks provided in the examples subfolder.

<div align="center">
  <img src="assets/ScannerInaccuracies.png" alt="Possible Scanner Inaccuracies" width="600" height="auto">
  <p><em>Some inaccuracies of a two axis scanner, which can be analyzed and corrected using SunscanPy</em></p>
</div>

## Installation
```bash
pip install sunscanpy
```

### From source

```bash
git clone https://github.com/Ockenfuss/sunscanpy.git
cd sunscanpy
pip install -e .
```

## Scanner Visualization
`sunscan/scanner_animation_streamlit.py` contains a streamlit application to visualize the orientation of the scanner, given a set of axis positions and scanner parameters. To run it:
```
pip install sunscanpy[dev] # needs streamlit, which is not installed by default with sunscanpy
streamlit run sunscan/scanner_animation_streamlit.py
```

## Contributing
See Contributing.md

## License

SunscanPy  © 2025 by Paul Ockenfuß, Gregor Köcher, Ludwig-Maximilians Universität München is licensed under CC BY-SA 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-sa/4.0/


## Authors

- Paul Ockenfuß
- Gregor Köcher
