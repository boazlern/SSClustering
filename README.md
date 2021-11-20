This is the code for the paper: ["Boosting The Performance of Semi Supervised Learning with Unsupervised Clustering".](https://arxiv.org/pdf/2012.00504.pdf)

To run the code, it is first recommended to create a new virtual environment and install all packages found in the attached requirements.txt file:

    python3 -m venv env
    source venv/bin/activate.csh
    pip install -r requirements.txt 
    
In the scripts folder, all the commands for running the paper's experiments can be found.  
For evaluation, the evaluation/evaluate.py can be used. All the different options are specified in the options.py file. For example, evaluating a model with its exponential moving average weights can be done with the command:

    python3 evaluation/evaluate.py --ckpt models/model.ckpt --ema
    
By default, the program uses all the gpus available. If specific gpus are desired, this can be requested with the --lab_gpu option. All desired gpus numbers need to be specified with a comma separator between them. 

Experimenting with a new clustering algorithm or a new semi-supervised algorithm is very easy. One can write the code to those algorithms in the algorithms folder and inherit from one of the existing classes. Then, the only thing needed is to add the name of the algorithm to the "S_ALGS" or "US_ALGS" in options.py and to the dictionary "s_algo_to_class" or "us_algo_to_class"  that is found in train.py 
