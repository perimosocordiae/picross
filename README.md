# Picross Puzzle Solver

## Install
```
python setup.py install
```

## Usage
```
# start the web UI
solve_picross.py --port 8765

# solve a specific puzzle
solve_picross.py "3,3,(3,1),5,(1,1,1)" "3,2,5,(2,1),5"
```

## Testing
```
python -m unittest discover
```
