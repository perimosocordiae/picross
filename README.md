# Picross Puzzle Solver

## Install
```
python setup.py install
```

### Dependencies

Not all dependencies are required for the basic command-line solver,
but full functionality will use:

 * numpy
 * scipy
 * scikit-learn
 * python-opencv
 * six

Finally, the web UI is powered by
[webtool](https://github.com/perimosocordiae/webtool),
which at the moment has to be installed from GitHub.

## Usage
```
# start the web UI
solve_picross.py --port 8765

# solve a specific puzzle
solve_picross.py "3,3,(3,1),5,(1,1,1)" "3,2,5,(2,1),5"

# detect and solve a puzzle from a screenshot image
solve_picross.py --image my_screenshot.png

# create an animation of the solution process
solve_picross.py --animate [other options]
```

## Testing
```
python -m unittest discover
```
