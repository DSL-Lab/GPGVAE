# Latent graph learning via VGAE
## Setup
```{bash}
conda create -n lgvae python=3.10
conda install -c dglteam/label/cu118 dgl
pip install -r requirements.txt
```

## Running the experiments

There are two steps to running the experiments:

- The first step is to run the classification.py file to use specformer to pre-train a classification model. After pre-training, the model is used to infer the prior label (correlation or anti-correlation) for each graph.

- The second step is to run the main.py to learn the graph structure based on VGAE.

For the pre-training stage, we have considered different methods, including specformer and simple MLP. The pre-trained models are provides in the folder 'data/model'. This model are trained on the synthetic datasets. Here we consider 5000 graphs with different type of games, different number of nodes and different number of games. At the second stage, we will first generate new datasets and then directly load the pre-trained models to infer the prior label for each new graph.

<!-- ### Synthetic Datasets

The base file for running all experiments is main.py. The arguments graph_type or game_type can be set to train the model on different scenarios.
For example, consider 50 barabasi_albert graphs with 20 nodes and 100 linear_quadratic games ($\beta = 0.8, \alpha = 0.5$).  Based the pre-trained Specformer model, run the following command:

```{bash}
python main.py --pre_model_type Specformer --n_graphs 50 --n_nodes 20 --n_games 100 --game_type linear_quadratic --alpha 0.5 --target_spectral_radius 0.8 --graph_type barabasi_albert
```

If you want to use the pre-trained MLP model, run the following command:

```{bash}
python main.py --pre_model_type MLP --n_graphs 50 --n_nodes 20 --n_games 100 --game_type linear_quadratic --alpha 0.5 --target_spectral_radius 0.8 --graph_type barabasi_albert
```

If you don't want to use pre-trained models, run the following command:

```{bash}
python main.py --pre_model_type Random --n_graphs 50 --n_nodes 20 --n_games 100 --game_type linear_quadratic --alpha 0.5 --target_spectral_radius 0.8 --graph_type barabasi_albert
``` -->

### Real World Datasets

#### Yelp Dataset
Download the Yelp pickle here:
https://drive.google.com/drive/folders/1QhArobPzsehf5PFZ_VzM12eJuyUKoqPi

it should lies in /data/Yelp/*.pickle

There are four yelp datasets in folder 'data/Yelp': PA_rating_food.pickle, PA_review_food.pickle, LA_rating_food.pickle and LA_review_food.pickle. We consider two types of data (rating and review) and two states (PA and LA). 

To run the experiments on the LA_rating_food.pickle dataset, run the following command:

```{bash}
 python main.py --pre_model_type Specformer --game_type Yelp --graph_type LA_rating --num_mix_component 1 --hidden_dim 128 --n_epochs 400 --n_heads 4 --lr 1e-4 --encoderA transformer --weight_decay 1e-3 --beta2 1 --beta1 10 --encoderZ mlp_on_seq --estimator reinforce --eta 0.99 --klA_nsample 50 --nll_nsample 1
```

Here,

'--num_mix_component' is the number of mixture distributions;

'--estimator' is the estimator for the gradient of the reconstruction loss, we can choose 'reinforce' or 'concrete';

'--eta' is the parameter for the reinforce estimator;

'--klA_nsample' is the number of samples for the KL divergence, this number cannot be too large, otherwise it will cause memory issue;

'--nll_nsample' is the number of samples for the reconstruction loss. Increasing this number will cause memory issue.

<!-- 
On Vector
```{bash}
scripts/submit.sh yelp n1 python gnn/main.py --pre_model_type Specformer --game_type realworld --graph_type yelp_rating --num_mix_component 1 --hidden_dim 32 --n_epochs 100 --lr 1e-4 --n_heads=4 --n_sample=20
``` -->
<!-- 
Muchen: There is two base encoder I'm playing with, use following command
```{bash}
# Transformer Encoder Capped at 0.6-0.7 Sometimes
python main.py --pre_model_type Specformer --game_type realworld --graph_type LA_rating --num_mix_component 2 --hidden_dim 64 --n_epochs 400 --n_heads 2 --n_sample 10 --lr 1e-4  --encoderA transformer --use_amp
``` -->
<!-- 
```{bash}
# MLP Encoder which I wrote and found can occasionaly reach 0.8 AUC
python main.py --pre_model_type Specformer --game_type realworld --graph_type LA_rating --num_mix_component 2 --hidden_dim 32 --n_epochs 400 --n_heads 2 --n_sample 10 --lr 1e-4  --encoderA mlp_encoder_2 --use_amp
``` -->


If you suspect there is a numerical issue, turn --use_amp off to debug in full precision



### Visualization

```bash
python gnn/vis.py --pre_model_type Random --n_graphs 50 --n_nodes 20 --n_games 100 --game_type linear_quadratic --graph_type barabasi_albert 
```
