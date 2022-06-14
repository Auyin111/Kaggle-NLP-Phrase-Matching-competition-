<details>
<summary>v4.0.0</summary>

___________________________________________________
  
  1. Training Data: 
  
    a. Support Google translate dataset augmentation (Only tried en + zh, not really working in the 1st try) 
  
    b. Group 2 context (short/medium/long) + mentioned groups (current not used, looking for the best way to use them) \n
  
  
  2. CustomModel:
  
    a. Multi Sampler Dropout
  
    b. Multi Head Self-Attention model head
  
    c. Weighted sum output of pretrained model layers
  
    d. Support 5 category output
  
  
  3. Loss function:
  
    BCE + BCEwithLogits + MSE + CCC1 + CCC2 (~CCC1 times training size) + PCC + Cross Entropy (for 5 category output only)
   
  
  4. Training/Optimizer:
  
    a. Stochastic weight average (swa) (Not really working ...)
  
    b. Cosine Annealing LR scheduler (For swa, but no warm-up available)
  
    c. Dynamic Padding (Improves training speed ~ 30-100%)
  
    d. Batch Sampler - by label or context (by context seems providing more stable training progress)
 
    
  5. Others
  
    a. Plot learning rate during training (Only useful for debug)
  
    b. Some basic conflict checking (eg: Not using cross entropy for 5 catergoy output model)
  
    c. Option for disabling model checking (Auto-disable when is_debug == True)
    
