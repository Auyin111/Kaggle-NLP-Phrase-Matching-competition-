<details>
<summary>Set up cloud and run main.py</summary>

_______________________________________


1. copy competition_patent and create competition_patent_upload (<mark>local cmd</mark>)
```commandline
python prepare_upload_folder.py
```

2. copy competition_patent_upload to cloud (<mark>local cmd</mark>)
```commandline
scp -r C:\Users\auyin11\PycharmProjects\competition_patent_upload USERNAME@IP:./
```

3. connect to server (<mark>local cmd</mark>)
```commandline
ssh USERNAME@IP
```

  4. setup paperspace cloud <mark>(the display driver is only for A4000 or A5000)</mark> or lambda gpu cloud
```commandline
bash competition_patent_upload/paperspace_setup.sh
```
```commandline
bash competition_patent_upload/lambda_labs_setup.sh
```

5. run main.py
```commandline
bash competition_patent_upload/run_main.sh
```
</details>
<br>

<details>
<summary>v3.1.1 (reference model)</summary>

___________________________________________________

- When to stop the training? 
  - train with all epoch and replace original model if Pearson correlation is better
- What is the original ensemble method? 
  - use 4 cross validation model to predict 4 score then average 4 score
- cv score in kaggle: 0.8101
</details>

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
  
    d. Saving the cfg.py and model.py used by the current version to the output directory right before training
  
  6. Add early stopping patience
  
</details>
