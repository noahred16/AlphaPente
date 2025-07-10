# AlphaPente
Pente AI agent inspired by AlphaZero

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


