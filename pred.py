import argparse
import os
import sys
import time
import shap
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from cgcnn.data1 import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal gated neural networks')
parser.add_argument('modelpath', help='path to the trained model.')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('file_name', help='id prop file name')
parser.add_argument('result_file_name', help='result file name')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

args = parser.parse_args(sys.argv[1:])
if os.path.isfile(args.modelpath):
    print(f"=> Loading model params '{args.modelpath}'")
    model_checkpoint = torch.load(args.modelpath, map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print(f"=> Loaded model params '{args.modelpath}'")
else:
    print(f"=> No model params found at '{args.modelpath}'")

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if model_args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.

def main():
    global args, model_args, best_mae_error

    # Load data
    dataset = CIFData(args.cifpath, args.file_name)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)

    # Build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=model_args.task == 'classification')
    if args.cuda:
        model.cuda()

    # Define loss function
    criterion = nn.NLLLoss() if model_args.task == 'classification' else nn.MSELoss()

    # Load checkpoint
    if os.path.isfile(args.modelpath):
        print(f"=> Loading model '{args.modelpath}'")
        checkpoint = torch.load(args.modelpath, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer = Normalizer(torch.zeros(3))
        normalizer.load_state_dict(checkpoint['normalizer'])
        print(f"=> Loaded model '{args.modelpath}' (epoch {checkpoint['epoch']}, validation {checkpoint['best_mae_error']})")
    else:
        print(f"=> No model found at '{args.modelpath}'")

    validate(test_loader, model, criterion, normalizer, test=True)

    # Run SHAP interpretability analysis
    interpret_with_shap(model, test_loader)

def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_args.task == 'regression':
        mae_errors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input[:4]  # Unpack crystal_atom_idx
            input_var = (
                atom_fea.cuda(non_blocking=True),
                nbr_fea.cuda(non_blocking=True),
                nbr_fea_idx.cuda(non_blocking=True),
                crystal_atom_idx.cuda(non_blocking=True)  # Pass crystal_atom_idx
            ) if args.cuda else (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            target_var = target.cuda(non_blocking=True) if args.cuda else target

        output = model(*input_var)  # Pass input_var including crystal_atom_idx
        loss = criterion(output, target_var)

        if model_args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Normalizer:
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))

def interpret_with_shap(model, data_loader):
    model.eval()
    data_iter = iter(data_loader)
    input, target, _ = next(data_iter)  # Get a batch
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input[:4]  # Include crystal_atom_idx

    # Convert PyTorch tensors to NumPy arrays
    atom_fea_np = atom_fea.detach().cpu().numpy()
    nbr_fea_np = nbr_fea.detach().cpu().numpy()
    nbr_fea_idx_np = nbr_fea_idx.detach().cpu().numpy()
    crystal_atom_idx_np = crystal_atom_idx.detach().cpu().numpy()

    # Combine inputs for SHAP
    data = (atom_fea_np, nbr_fea_np, nbr_fea_idx_np, crystal_atom_idx_np)

    # Define prediction function for SHAP
    def model_predict(data):
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = data
        atom_fea = torch.tensor(atom_fea).float()
        nbr_fea = torch.tensor(nbr_fea).float()
        nbr_fea_idx = torch.tensor(nbr_fea_idx).long()
        crystal_atom_idx = torch.tensor(crystal_atom_idx).long()

        input_var = (
            atom_fea.cuda(),
            nbr_fea.cuda(),
            nbr_fea_idx.cuda(),
            crystal_atom_idx.cuda()
        )
        with torch.no_grad():
            output = model(*input_var)
        return output.cpu().numpy()

    # Initialize SHAP explainer
    explainer = shap.KernelExplainer(model_predict, data)

    # Compute SHAP values for the first sample
    shap_values = explainer.shap_values([atom_fea_np[0], nbr_fea_np[0], nbr_fea_idx_np[0], crystal_atom_idx_np[0]])

    # Visualize SHAP values
    shap.summary_plot(shap_values, atom_fea_np)

if __name__ == '__main__':
    main()
