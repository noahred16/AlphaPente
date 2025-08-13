# AlphaPente
Pente AI agent inspired by AlphaZero

## Process
Generate training data through self play using a NN backed MCTS algorithm with specified number of simulations.
```
python generate.py --simulations 100
```


Check db
```
python query.py
```

Do training
```
python train.py
```





## AlphaZero
AlphaZero incrementally learning through self play through a NN backed MCTS algorithm. 
It uses a NN to predict the policy and value for each state. 
As the algorithm continues, it learns to play better and better games. 
Through such, it generates more accurate estimates for the policy and value of the game states. 
The insights are generated through the process of the MCTS.

Works through a NN MCTS

## Quickstart
<!-- start python virtual env -->
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

<!-- generate requirments.txt -->
pip freeze > requirements.txt



## GUI
flask --app play run --debug




## TODO
Webapp:
- allow white and black to be played by different players
- better state management
- better ai

Training:
- each position could be flipped and rotated to increase the training set (total of 16 positions)

Generation:
- anayze the bottleneck. each game takes a long time to generate
- probably its the gpu part, I'd think. 
- combine value and policy prediction into a batch call. 
- don't call the model if the node already has a policy and value


Generate games in parallel using multiprocessing. 
- use a queuing system to write the games to the db table for storage.


