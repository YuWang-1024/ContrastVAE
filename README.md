This is the repository for paper: ContrastVAE: Contrastive Variational AutoEncoder for Sequential
Recommendation

## the model version control:

--variational_dropout : using the variational augmentation 

--latent_contrastive_learning: using the model augmentation

--latent_data_augmentation: using the data augmentation

--VAandDA: using both variational augmentation and data augmentation 

without any above version control: the model is the vanilla attentive variational autoencoder


## train on Beauty
python main.py --latent_contrastive_learning --data_name=Beauty --latent_clr_weight=0.6 --reparam_dropout_rate=0.1 --lr=0.001 --hidden_size=128 --max_seq_length=50 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRec --attention_probs_dropout_prob=0.0 --anneal_cap=0.2 --total_annealing_step=10000

## Office
python main.py --variational_dropout --gpu_id 1 --data_name=Office_Products --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRec --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=20000 

## Tool
python main.py --variational_dropout --gpu_id 1 --data_name=Tools_and_Home_Improvement --latent_clr_weight=0.4 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.4 --total_annealing_step=5000 

## Toy

python main.py --variational_dropout --gpu_id 1 --data_name=Toys_and_Games --latent_clr_weight=0.3 --lr=0.001 --hidden_size=128 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --model_name=VAGRecVD --attention_probs_dropout_prob=0.3 --anneal_cap=0.2 --total_annealing_step=10000 

