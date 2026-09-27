import numpy as np
from ray import tune

def create_tuning_params(run_type='single', seed=1):
  
    if (run_type == 'single') | (run_type == 'full'):
        params = {
            'num_runs':1,

            'input':
            {
                'num_epochs':1000,
                'dropout':0.4,
                'batch_size':64,
                'learning_rate':1e-5,
                'steps':[11, 12],
                'warmup':0,
                'betas':(0.9, 0.999),
                'weight_decay':0.05,
                'loss_type':'CE',
                'fl_beta':0.9999,
                'fl_gamma':2
            }
        }
    
    elif run_type == 'optimize':
        params = {
            'num_runs': 20,

            'input':
            {
                'num_epochs':10,
                'dropout':0.2,
                'batch_size':64,
                'learning_rate':1e-5,
                'steps':[11, 12],
                'warmup':0,
                'betas':(0.9, 0.999),
                'weight_decay':1e-3,
                'loss_type':'CE',
                'fl_beta':0.9999,
                'fl_gamma':3
            },
            
            'tune':
            {
                'num_epochs':tune.choice([10]),
                'dropout':tune.choice([0.2, 0.3, 0.4, 0.5]),
                'batch_size':tune.choice([32, 64]),
                'learning_rate':tune.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]), # Note: Enet larger learning rate didn't work well with FL, added 1e-5 (5e-6, 1e-5, 5e-5, 1e-4)
                'steps':tune.choice([[11, 12]]),
                'warmup':tune.choice([0]),
                'betas':tune.choice([(0.9, 0.9999)]),
                'weight_decay':tune.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]), # Note: Enet smaller weight decay didn't work well FL, added 1e-2 (1e-5, 1e-4, 1e-3, 1e-2)
                'loss_type':tune.choice(['CE']),
                'fl_beta':tune.choice([0.9999]),
                'fl_gamma':tune.choice([3])
            },
        }

    elif run_type == 'random':
        rnd = np.random.RandomState(seed)

        params = {
            'num_runs':20,

            'input':
            {
                'num_epochs':10,
                'dropout':0.2,
                'batch_size':64,
                'learning_rate':1e-5,
                'steps':[11, 12],
                'warmup':0,
                'betas':(0.9, 0.999),
                'weight_decay':1e-3,
                'loss_type':'CE',
                'fl_beta':0.9999,
                'fl_gamma':3
            },
            
            'tune':
            {
                'num_epochs':10,
                'dropout':rnd.choice([0.2, 0.3, 0.4, 0.5]),
                'batch_size':rnd.choice([32, 64]),
                'learning_rate':rnd.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]), #round(rnd.uniform(low=5e-6, high=1e-1), 8)
                'steps':[11, 12], #rnd.choice([[11, 20]])
                'warmup':0,
                'betas':(0.9, 0.999), #(round(rnd.uniform(low=0.9, high=0.999),3), round(rnd.uniform(low=0.9, high=0.999),3))
                'weight_decay':rnd.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]), #round(rnd.uniform(low=5e-6, high=1e-1), 8)
                'loss_type':'CE', #rnd.choice(['CE','FL'])
                'fl_beta':0.9999, #round(rnd.uniform(low=0.9, high=0.9999),4)
                'fl_gamma':2 #rnd.choice([2,3,4,5])
            }
        }
    
    elif run_type == 'test':

        params = {
            'num_runs':1,

            'input':
            {
                'num_epochs':10,
                'dropout':0.4,
                'batch_size':64,
                'learning_rate':1e-5,
                'steps':[11, 12],
                'warmup':0,
                'betas':(0.9, 0.999),
                'weight_decay':0.05,
                'loss_type':'CE',
                'fl_beta':0.9999,
                'fl_gamma':2
            }
        }
    
    elif run_type == 'visualize':

        params = {
            'num_runs':1,

            'input':
            {
                'num_epochs':10,
                'dropout':0.4,
                'batch_size':64,
                'learning_rate':1e-5,
                'steps':[11, 12],
                'warmup':0,
                'betas':(0.9, 0.999),
                'weight_decay':0.05,
                'loss_type':'CE',
                'fl_beta':0.9999,
                'fl_gamma':2
            }
        }
    
    return params