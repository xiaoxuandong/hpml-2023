import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
from model import build_model
from datasets import get_datasets, get_data_loaders
from utils import save_model, save_plots
import time
import numpy as np
import torch.optim.lr_scheduler
from torch.profiler import profile, record_function, ProfilerActivity
# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=30,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-pt', '--pretrained', action='store_true',
    help='Whether to use pretrained weights or not'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.001,
    help='Learning rate for training the model'
)
parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
parser.add_argument('--cpu', dest='cuda', action='store_false', help='use cpu')
parser.add_argument('--num_workers', default=4, type=int, help='number of data load workers')
parser.add_argument('--batch_size', default=16, type=int, help='batch_size')

args = vars(parser.parse_args())
print(args)

# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    data_time = 0.0 
    start = perf_counter()
    end = perf_counter()
    for i, data in enumerate(trainloader):

        # measure data loading time
        data_time += perf_counter() - end
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
        end = perf_counter()

    total_time = perf_counter() - start
    print('Data-loading time: %.3f s, Training time: %.3f s, Total time: %.3f s' % (
            data_time, total_time - data_time, total_time))
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc, data_time, total_time
# Validation function.
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    total_time = 0.0

    data_time = 0.0 
    start = perf_counter()
    end = perf_counter()
    with torch.no_grad():
        for i, data in enumerate(testloader):

            # measure data loading time
            data_time += perf_counter() - end

            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()


            end = perf_counter()

    total_time = perf_counter() - start
    print('Data-loading time: %.3f s, Training time: %.3f s, Total time: %.3f s' % (
            data_time, total_time - data_time, total_time))
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc, data_time, total_time

if __name__ == '__main__':
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets(args['pretrained'])
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid,batch_size=args['batch_size'], num_workers=args['num_workers'])
    # Learning_parameters. 
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if args['cuda'] else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    model = build_model(
        pretrained=args['pretrained'], 
        fine_tune=True, 
        num_classes=len(dataset_classes)
    ).to(device)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    if args['optimizer']  == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args['learning_rate'])
    elif args['optimizer']  == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), rho=0.9)
    elif args['optimizer']  == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    elif args['optimizer']  == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9, weight_decay=5e-4)   
    elif args['optimizer']  == 'SGD_Nesterov':
        optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9, nesterov=True)

    # # scheduler
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []


    total_data_time = []
    total_time = []
    i_data_time = []
    i_time = []


    # Start the training.
    for epoch in range(epochs):

        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc, t1, t2 = train(model, train_loader, 
                                                optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc, t3, t4 = validate(model, valid_loader,  
                                                    criterion)
        total_data_time.append(t1)
        total_time.append(t2)
        i_data_time.append(t3)
        i_time.append(t4)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)

    print('total_data_time', total_data_time , 'total_time', total_time)
    print('Avg data time:', np.mean(total_data_time), ',total_time:', np.mean(total_time))

    print('Avg inference data time:', np.mean(i_data_time), ',inference total time:', np.mean(i_time))

    # # Compute FLOPs per second
    # input_size = np.prod(next(iter(train_loader))[0].size()[1:])
    # n_flops = np.prod(input_size) * model.classifier[1].in_features * 2 # multiply by 2 for multiply and add operations
    # flops_per_sec = n_flops * len(train_loss) * epochs / total_time

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            validate(model, valid_loader, criterion)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        
    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, args['pretrained'])
    print('Train_loss:', train_loss)
    print('Train_acc:', train_acc)
    print('Valid_loss:', valid_loss)
    print('Valid_acc:', valid_acc)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, args['pretrained'])
    print('TRAINING COMPLETE')

