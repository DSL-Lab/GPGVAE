# Learning Latent Structures in Network Games via Data-Dependent Gated-Prior Graph Variational Autoencoders

This is the official implementation of [GPGVAE]() in PyTorch.

   
## Install dependencies
```{bash}
conda create -n gpgvae python=3.10
conda activate gpgvae
pip install dgl==1.0.1+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html
pip install -r requirements.txt
```

## Datasets
### Synthetic Datasets
The base file for running all experiments is **main.py**. The arguments **graph_type** or **game_type** can be set to train the model on different scenarios.
For example, consider a barabasi_albert graph with 20 nodes and 100 linear_quadratic games ($\beta = 0.8, \alpha = 0.5$).  Based the pre-trained Specformer model, run the following command:

```{bash}
python main.py --pre_model_type Specformer --n_nodes 20 --n_games 100 --game_type linear_quadratic --alpha 0.5 --target_spectral_radius 0.8 --graph_type barabasi_albert --eta 0.99 --n_epochs 200
```

### Real-world Datasets
The Indian Villages datasets and two Foursquare datasets are provided in folders '/data/indian_networks/' and '/data/Foursquare/'.


To run the experiments on the NYC_dataset.pickle dataset, run the following command:

```{bash}
python main.py --pre_model_type Specformer --game_type Foursquare --graph_type NYC --hidden_dim 256 --n_heads 4 --lr 1e-4 --encoderA transformer --weight_decay 1e-3 --beta1 10 --encoderZ mlp_on_seq --klA_nsample 30 --nll_nsample 1 --num_mix_component 3 --estimator reinforce --eta 0.97 --n_epochs 200
```

To run the experiments on the Indian Villages datasets, run the following command:

```{bash}
python main.py --pre_model_type Specformer --game_type village --graph_type indian_village --hidden_dim 128 --n_heads 4 --lr 1e-4 --encoderA transformer --weight_decay 1e-3 --beta1 10 --encoderZ mlp_on_seq --klA_nsample 1 --nll_nsample 1 --num_mix_component 1 --estimator reinforce --eta 0.99  --n_epochs 100 --batch_size 1
```

You can download the Yelp datasets here:
https://drive.google.com/drive/folders/1QhArobPzsehf5PFZ_VzM12eJuyUKoqPi

it should lies in /data/Yelp/*.pickle.

There are four yelp datasets in folder 'data/Yelp': PA_rating_food.pickle, PA_review_food.pickle, LA_rating_food.pickle and LA_review_food.pickle. We consider two types of data (rating and review) and two states (PA and LA). 

To run the experiments on the LA_review_food.pickle dataset, run the following command:

```{bash}
python main.py --pre_model_type Specformer --game_type Yelp --graph_type LA_review --hidden_dim 128 --n_heads 4 --lr 1e-4 --encoderA transformer --weight_decay 1e-3 --beta1 10 --encoderZ mlp_on_seq --klA_nsample 1 --nll_nsample 1 --num_mix_component 1 --estimator reinforce --eta 0.99  --n_epochs 600
```

Here,

'--pre_model_type' is the pre-trained specformer interaction encoder at the first stage.

'--num_mix_component' is the number of mixture distributions;

'--estimator' is the estimator for the gradient of the reconstruction loss, we can choose 'reinforce' or 'concrete';

'--eta' is the parameter for the reinforce estimator;

'--klA_nsample' is the number of samples for the KL divergence, this number cannot be too large, otherwise it will cause memory issue;

'--nll_nsample' is the number of samples for the reconstruction loss. 


## Citation [TBA]

If you find our work is useful, please cite the [paper]():

    Xue Yu, Muchen Li, Yan Leng, and Renjie Liao. Learning Latent Structures in Network Games via Data-Dependent Gated-Prior Graph Variational Autoencoders. ICML 2024.

## Contact

Please submit a Github issue or contact xueyu_2019@ruc.edu.cn if you have any questions or find any bugs.