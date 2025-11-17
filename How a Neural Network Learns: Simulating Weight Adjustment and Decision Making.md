# Neural Network Learning Simulation

## Step 1: Inputs → Decision → Output
- **Inputs:** Food, Service, Price  
- **Middle Node:** Decision Node  
- **Output:** Like / Dislike  

## Step 2: Roles
- **Input Feeder:** Reads restaurant scenario  
- **Decision Node:** Predicts Like/Dislike  
- **Observers:** Track results and adjust tokens (weights)  

## Step 3: Training Phase (10 Scenarios)
- Started with equal token weights (**1 each**).  
- Each incorrect prediction triggered weight increases on the features that actually mattered.  
- Accuracy improved as weights shifted.

### Example Table

| Round | Food | Service | Price | Prediction | Correct? | Notes |
|-------|------|----------|--------|-------------|-----------|--------|
| 1 | Like | — | — | Like | No | Underestimated price impact |
| 5 | Like | — | — | Like | Yes | Food dominates |
| 10 | Like | — | — | Like | Yes | Learned that good food > price |

## Step 4: Testing Phase (3 New Scenarios)
Used final weights from training to make predictions.  
**Result: 2/3 correct** — shows the model generalized instead of memorizing.

## Step 5: Reflection
- Predictions improved once tokens (weights) were adjusted — that’s the entire learning mechanism.  
- **Food** ended with the strongest weight, meaning it influenced decisions the most.  
- This behavior mirrors real neural networks: weights shift to reduce error.  
- More layers/inputs = more capability, slower training.
