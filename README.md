# Undersrtanding-Dynamic-Scenes-using-MR-GCN

For any queries mail : saisravan.m@research.iiit.ac.in (or)
mahtab.sandhu@gmail.com

Our method detects and classifies objects of interest (vehicles) in a video scene into 6 classes moving away, moving towards us, parked, lane change(L->R), lane change(R->L), overtake. 

**Note that our method is not based on classifying the ego-vehicle.**

---------------------
### Video GIF

<img src="cut_video.gif?raw=true">

### Dataset
-----------
We selected 4 main datasets to perform the experiments.
1. [Apollo](http://apolloscape.auto/scene.html) 
2. [Kitti](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)
3. [Honda](https://usa.honda-ri.com/H3D)
4. Indian


**Graphs for all datasets can be downloaded from [graphs](https://drive.google.com/drive/folders/1Jm3oQr-05VDTnybUakRXL-fxBXDQ5VtG?usp=sharing).<br />
For information on how each graph is stored as a npz file, go through the README file in the same link.**

On apollo we have selected sequences from scene-parsing dataset and picked around 70 small sequences(each containing around 100 images) manually that include behaviors of our interest. In Honda, we have taken sequences that include various number of lane change scenarios from their H3D Dataset. Similarly on Kitti, we use tracking sequences 4,5,10 which are in line with our class requirement.

### Installation
--------------
##### Requirements
```
dgl
pytorch == 1.2.0
pandas
numpy
tqdm
```

#### Installing without GPU:
```
pip3 install requirements.txt
```
To install and use GPU for dgl, cuda support can be installed from their official website, [dgl](https://www.dgl.ai/pages/start.html) .<br /> 
And set *use_cuda = 1* in training/testing codes.

## Training and Testing on Apollo dataset 
```
git clone https://github.com/ma8sa/Undersrtanding-Dynamic-Scenes-using-MR-GCN.git
Undersrtanding-Dynamic-Scenes-using-MR-GCN

--Relational attention
python3 rel-att-gcn_test.py			   # for testing trained model
python3 rel-att-gcn_train.py		   # for training the complete model

--MRGCN
python3 MRGCN_test_apollo.py		   # for testing trained model
python3 MRGCN_train_apollo.py		   # for training the complete model

---MRGCN+LSTM
python3 lstm_rgcn_test_apollo.py       # for testing trained model
python3 lstm_rgcn_train_apollo.py      # for training the complete model

```
**NOTE** : Make sure to extract the corresponding graphs (*graphs_apollo*) and place it in the same folder where you are running the code from.

In training, *rel-att-gcn_train.py* has all the parameters to tune. *main_model.py* contains the complete model. *rgcn_layer.py* contains the MR-GCN layer implemented using attention. *graphs_preproc_apollo.py* conatains all the data preprocessing methods used. 

## Testing on Honda/Indian/Kitti dataset (Transfer Learning)
```
for Honda,
python3 transfer_testing.py Honda
for indian,
python3 transfer_testing.py indian
for kitti,
python3 transfer_testing.py kitti
```
**NOTE** : Make sure to extract the corresponding graphs (*graphs_kitti* for **kitti** and *graphs_indian* for **indian** and *graphs_honda* for **Honda**) and place it in the same folder where you are running the *transfer_testing.py* code from.
### RESULTS
---------
0->Move forward<br />
1->Moving towards us<br />
2->Parked<br />
3-> lane-change(L->R)<br />
4-> lane-change(R->L)<br />
5-> Overtake

##### Results on Apollo <br> Rel-Attentive GCN
|  | 0 | 1 | 2 | 3 | 4 | 5 |
| ------------- | ------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| class accuracy(train)| 95 | 98 | 98 | 95 | 96 | 88 |  
| class counts(train)  | 2667 | 621 | 3357 |393  | 428 | 525  |
| class accuracy(val)  | 95 | 99 | 98 | 97 | 96 | 89 |
| class counts(val)  | 814 | 237 | 1400 | 162 | 130 | 73 |

##### Results on Honda tested with weights trained on Apollo
|  | 0 | 1 | 2 | 3 | 4 |
| ------------- | ------------- | ------------ | ------------ | ------------ | ------------ |
| class accuracy| 92 | 92 | 99 | 94 | 92 |  
| class counts  | 445 | 359 | 1237 | 229 | 114 |

##### Results on Kitti tested with weights trained on Apollo
|  | 0 | 1 | 2 |
| ------------- | ------------- | ------------ | ------------ |
| class accuracy| 99 | 98 | 98 |
| class counts  | 504 | 230 | 674 |

##### Results on Indian tested with weights trained on Apollo
|  | 0 | 1 | 2 |
| ------------- | ------------- | ------------ | ------------ |
| class accuracy| 99 | 92 | 99 |
| class counts  | 324 | 229 | 2547 |


---------------------
### Base-line Implementation details (St-RNN)
-------------
We provide comparison with with Structural-RNN, a LSTM based graph network. Since the tasks in their paper confine only to driver-anticipation, we use one of their methods similar to our task. Specifically, we use the **detection** method of **activity-anticipation** mentioned in the paper due to the similarity in the architecture and task . We use *Vehicles* as *Humans* and *Lane Markings* as *Objects* in their architecture for our purpose. Similar to the Human-Object, Human-Human and Object-Object interactions, we observe the Vehicle-Lane, Vehicle-Vehicle and Lane-Lane interactions for all time-steps as it takes input for each time-step and for each possible relation.

Different embeddings are given based on nature of object (car/lane). The embeddings are taken from our MRGCN-LSTM trained model (on Apollo). As St-RNN expects input for each object for every relation, we use zeros in case an object is not involved in relation *r*. Hence each object has an embedding for each relation whether or not it exhibits such relation. 

Our model outperforms it due to one main reason, lack of sophisticated graphical structure as in MR-GCN where information is based only on relations a node exhibits.  


| Method  | St-RNN | MRGCN +<br> LSTM | MRGCN | Rel-Att-GCN | 
| ------------- | ------------ | ------------| -------------| -------------|
| Moving away  | 76  |	85 | 94 | 95 |
| moving towards us  | 51 |	89 | 95 | 99 |
| Parked  | 83  | 94 | 94 | 98 |
| lane-change(L->R)  | 52  | 84 | 97 | 97 |
| lane-change(R->L)  | 57  | 86 | 93 | 96 |
| Overtake  | 63  | 72 | 86 | 89 |



<!---
<img src="cut_video.gif?raw=true">
>
### Attention Explanantion
-----------
Due to space constraint in the paper, we have defined attention as a module in the paper. Here, we give it's working and explanation.<br/>
To weight the outputs from LSTM(which are ordered w.r.t time), we use attention as a weighted sum for predicting the output.<br/>
>
Given output from LSTM as L<sub>g</sub>,
we define a HEAD as triplet containing Query(Q),Key(K),Value(V). The query, Key and Values are learnable intermediate parameters. Q and K are used to find which values of input are similar/highly related and V is to weight them. Hence, the equation becomes : 
>
![attention_eqn](https://drive.google.com/uc?export=view&id=1AsejV-js_mxJ3oJnoLqMDZwBGRBrgj0B)
>
dk is the sacling factor(from paper). This is applied for all time-stamps.<br/> 
As dimension of L<sub>g</sub> is N x T x d<sub>2</sub>, attention using Q,K,V on **each node** gives, T x d<sub>3</sub> output. **Attention applies the above equation for all time-stamps, hence the T x d<sub>3</sub> output**.<br/>
If h heads are available, all heads are concatenated not across time but across d<sub>3</sub> dimension. Hence, output dimension remains same as T x d<sub>3</sub>, as we finally project to input dimension for output from attention.
![mh eqn](https://drive.google.com/uc?export=view&id=1RGs2zFIPcZA6t3jTy0S07BM-c_6rG3jQ)
>
where head<sub>i</sub> = Attention(Q,K<sub>i</sub>,V<sub>i</sub>).<br/>
The final out put of attention is T x d<sub>3</sub> for **each node**.
--->
