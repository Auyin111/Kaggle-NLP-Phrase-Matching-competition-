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

4. setup paperspace cloud or lambda gpu cloud
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
