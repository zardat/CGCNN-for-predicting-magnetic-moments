import argparse, os, shutil, sys, time, warnings, csv
from random import sample
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from cgcnn.data1 import CIFData, collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

# Parse arguments for model configuration
parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+', help='Dataset options, starting with root dir path')
parser.add_argument('file_name', metavar='OPTIONS', nargs='+', default='id prop.csv', help='Input data file name')
parser.add_argument('result_file_name', metavar='OPTIONS', nargs='+', default='test_results.csv', help='Output file name')
parser.add_argument('--task', choices=['regression', 'classification'], default='regression', help='Task type')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, help='Data loading workers (default: 0)')
parser.add_argument('--epochs', default=100, type=int, help='Number of total epochs (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, help='Manual epoch number (for restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, help='Mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate (default: 0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int, help='Scheduler milestones (default: [100])')
parser.add_argument('--momentum', default=0.9, type=float, help='Optimizer momentum')
parser.add_argument('--weight-decay', default=0, type=float, help='Weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='Print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, help='Path to latest checkpoint')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, help='Train data ratio (default: None)')
train_group.add_argument('--train-size', default=None, type=int, help='Train data size (default: None)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, help='Validation data ratio (default: 0.1)')
valid_group.add_argument('--val-size', default=None, type=int, help='Validation data size (default: 1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, help='Test data ratio (default: 0.1)')
test_group.add_argument('--test-size', default=None, type=int, help='Test data size (default: 1000)')
parser.add_argument('--optim', default='SGD', type=str, help='Optimizer type (SGD/Adam)')
parser.add_argument('--atom-fea-len', default=64, type=int, help='Hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, help='Hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, help='Number of convolution layers')
parser.add_argument('--n-h', default=1, type=int, help='Number of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])
best_mae_error = 1e10 if args.task == 'regression' else 0.0  # Set MAE threshold based on task

# Main training function
def main():
    global args, best_mae_error
    dataset = CIFData(*args.data_options, *args.file_name)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset, collate_fn=collate_fn, batch_size=args.batch_size,
        train_ratio=args.train_ratio, num_workers=args.workers, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, pin_memory=False, train_size=args.train_size,
        val_size=args.val_size, test_size=args.test_size, return_test=True
    )

    # Normalizer setup for regression task
# Normalizer setup for regression task
    if args.task == 'regression':
    # Use the entire dataset for calculating the normalizer
        all_data_list = [dataset[i] for i in range(len(dataset))]
        _, all_target, _ = collate_pool(all_data_list)
        normalizer = Normalizer(all_target)
    else:
        normalizer = None

    # Model setup
    structures, _, _ = dataset[0]
    orig_atom_fea_len, nbr_fea_len = structures[0].shape[-1], structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, args.atom_fea_len, args.n_conv, args.h_fea_len, args.n_h, classification=False)

    # Define criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # Load from checkpoint if available
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        args.start_epoch, best_mae_error = checkpoint['epoch'], checkpoint['best_mae_error']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if normalizer:
            normalizer.load_state_dict(checkpoint['normalizer'])
        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    # Train and validate for each epoch
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, normalizer)
        mae_error = validate(val_loader, model, criterion, normalizer)
        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)
        scheduler.step()
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1, 'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error, 'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict() if normalizer else None, 'args': vars(args)
        }, is_best)

    # Test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(best_checkpoint['state_dict'])
    test(test_loader, model, criterion, normalizer)  # Call the test function here

# Training function
def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    model.train()
    batch_time, data_time, losses, mae_errors = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()

    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_var = (Variable(input[0]), Variable(input[1]), input[2], [crys_idx for crys_idx in input[3]])
        target_normed = normalizer.norm(target) if normalizer else target
        target_var = Variable(target_normed)

        output = model(*input_var)
        loss = criterion(output, target_var)
        mae_error = mae(normalizer.denorm(output.data), target) if normalizer else mae(output.data, target)

        losses.update(loss.data, target.size(0))
        mae_errors.update(mae_error, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}] Time {batch_time.val:.3f} ({batch_time.avg:.3f}) Loss {losses.val:.4f} ({losses.avg:.4f}) MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})')

# Validation function
def validate(val_loader, model, criterion, normalizer):
    model.eval()
    batch_time, losses, mae_errors = AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()

    for i, (input, target, _) in enumerate(val_loader):
        input_var = (Variable(input[0]), Variable(input[1]), input[2], [crys_idx for crys_idx in input[3]])
        target_normed = normalizer.norm(target) if normalizer else target
        target_var = Variable(target_normed)

        with torch.no_grad():
            output = model(*input_var)
            loss = criterion(output, target_var)
            mae_error = mae(normalizer.denorm(output.data), target) if normalizer else mae(output.data, target)

        losses.update(loss.data, target.size(0))
        mae_errors.update(mae_error, target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Validate: [{i}/{len(val_loader)}] Time {batch_time.val:.3f} ({batch_time.avg:.3f}) Loss {losses.val:.4f} ({losses.avg:.4f}) MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})')

    print(f'* Validation MAE: {mae_errors.avg:.3f}')
    return mae_errors.avg


# Testing function
def test(test_loader, model, criterion, normalizer):
    model.eval()
    batch_time, losses, mae_errors = AverageMeter(), AverageMeter(), AverageMeter()
    test_preds, test_targets, test_cif_ids = [], [], []
    end = time.time()

    for i, (input, target, batch_cif_ids) in enumerate(test_loader):
        input_var = (Variable(input[0]), Variable(input[1]), input[2], [crys_idx for crys_idx in input[3]])
        target_normed = normalizer.norm(target) if normalizer else target
        target_var = Variable(target_normed)

        with torch.no_grad():
            output = model(*input_var)
            loss = criterion(output, target_var)
            mae_error = mae(normalizer.denorm(output.data), target) if normalizer else mae(output.data, target)

        losses.update(loss.data, target.size(0))
        mae_errors.update(mae_error, target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Collect predictions and targets for correlation calculation and plotting
        test_pred = normalizer.denorm(output.data) if normalizer else output.data
        test_preds += test_pred.view(-1).tolist()
        test_targets += target.view(-1).tolist()
        test_cif_ids += batch_cif_ids

        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(test_loader)}] Time {batch_time.val:.3f} ({batch_time.avg:.3f}) Loss {losses.val:.4f} ({losses.avg:.4f}) MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})')

    # Save results to CSV
    with open(args.result_file_name[0], 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['cif_id', 'target', 'predict'])
        writer.writerows(zip(test_cif_ids, test_targets, test_preds))

    # Calculate and print r² score and Pearson correlation coefficient (r)
        r2 = r2_score(test_targets, test_preds)
        pearson_corr, _ = pearsonr(test_targets, test_preds)
        print(f'R² Score: {r2:.4f}\nPearson Correlation Coefficient: {pearson_corr:.4f}')


# Create a larger figure with specified size and high DPI
    plt.figure(figsize=(6, 4), dpi=300)  # Increased figure size

# Scatter plot of actual vs predicted values with larger markers
    plt.scatter(test_targets, test_preds, alpha=0.5, s=20, color='blue')  # Larger markers with edge color

# Plotting the reference line
    plt.plot([0.0,0.25], [0.0,0.25], 'r--', lw=1)
    plt.xlim(-0.01, 0.25)
    plt.ylim(-0.01, 0.25)
# Set labels with increased font sizes
    plt.xlabel('True Magnetization', fontsize=14)  # Increased font size
    plt.ylabel('Predicted Magnetization', fontsize=14)  # Increased font size

# Set title with increased font size
    plt.title('DST-3 LOEO V FM w/o elemental magmom ', fontsize=15)  # Increased font size

# Enable grid for better readability
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()

# Save the figure in high quality (consider saving as SVG)
    plt.savefig('dst3_vfm_nomm.png', dpi=300)
         
    print(f'* Test MAE: {mae_errors.avg:.3f}')
    return mae_errors.avg

# Save model checkpoint
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# Metrics helpers
class Normalizer:
    def __init__(self, tensor):
        self.mean, self.std = torch.mean(tensor), torch.std(tensor)

    def norm(self, tensor): return (tensor - self.mean) / self.std
    def denorm(self, normed_tensor): return normed_tensor * self.std + self.mean
    def state_dict(self): return {'mean': self.mean, 'std': self.std}
    def load_state_dict(self, state_dict): self.mean, self.std = state_dict['mean'], state_dict['std']

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1): self.val, self.sum, self.count = val, self.sum + val * n, self.count + n; self.avg = self.sum / self.count

def mae(pred, target): return torch.mean(torch.abs(pred - target))

if __name__ == '__main__':
    main()