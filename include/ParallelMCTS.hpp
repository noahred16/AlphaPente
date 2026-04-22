/*
Notes (DO NOT DELETE!):

SolvedStatus
SearchMode
HeuristicMode

Config - should look the same. 

Node - should have same props but lets call it ThreadSafeNode. 

Additions: 
EvaluationQueue
WorkerPool
VirtualLossManager

WORKER THREAD PSEUDO-CODE
# we'll need to keep track of the ones in progress as well so that we don't over shoot the number of iterations
evaluationQueue = []
backpropagationQueue = []

numOfIterations = 1000
n_in_progress = 0
n_complete = 0
while n_in_progress < numOfIterations or n_complete < numOfIterations:
    if n_in_progress < numOfIterations: 
        n_in_progress += 1
        v = select(root)
        evaluationQueue.push(v)
    
    
    if backpropagationQueue.size() > 0
        v, p = backpropagationQueue.pop()
        value = v
        policy = p
        expand(v, policy)
        backpropagate(value)
        n_complete += 1


EVALUATION THREAD PSEUDO-CODE
batch_size = 1
while true:
    batch = []
    for i in range(batch_size):
        if evaluationQueue.size() > 0:
            batch.push(evaluationQueue.pop())
    
    if batch.size() > 0:
        values, policies = evaluate(batch)
        for v, p in zip(values, policies):
            backpropagationQueue.push((v, p))

*/