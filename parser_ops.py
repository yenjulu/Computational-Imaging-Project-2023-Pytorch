import argparse

def get_parser():

    parser = argparse.ArgumentParser(description="SSDU: Self-Supervision via Data Undersampling")
    parser.add_argument("--config", type=str, required=False, default="configs/base_modl,k=10.yaml",
                        help="config file path")
    parser.add_argument("--workspace", type=str, default='./workspace')
    parser.add_argument("--final_report", type=str, default='./final_report')
    parser.add_argument("--tensorboard_dir", type=str, default='./runs')
    parser.add_argument("--save_step", type=int, default=1)
    parser.add_argument("--write_lr", type=bool, default=False)
    parser.add_argument("--write_image", type=int, default=0)
    parser.add_argument("--write_lambda", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1)

    # %% hyperparameters for the unrolled network
    parser.add_argument('--acc_rate', type=int, default=8,            #  need to set 
                        help='acceleration rate')
    parser.add_argument('--CG_Iter', type=int, default=10,          #  need to set default=10
                        help='number of Conjugate Gradient iterations for DC')

    # %% hyperparameters for the dataset
    parser.add_argument('--nrow_GLOB', type=int, default=396,
                        help='number of rows of the slices in the dataset')
    parser.add_argument('--ncol_GLOB', type=int, default=768,
                        help='number of columns of the slices in the dataset')
    parser.add_argument('--ncoil_GLOB', type=int, default=16,
                        help='number of coils of the slices in the dataset')

    # %% hyperparameters for the SSDU
    parser.add_argument('--mask_type', type=str, default='Gaussian',
                        help='mask selection for training and loss masks', choices=['Gaussian', 'Uniform'])
    parser.add_argument('--rho', type=float, default=0.4,
                        help='cardinality of the loss mask, \ rho = |\ Lambda| / |\ Omega|')


    return parser
