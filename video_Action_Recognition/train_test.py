import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:true"
import torch
from torch.utils.data import DataLoader , random_split
from utils import plot_sequences , plot_training_validation_loss_accuracy , plot_confusion_matrix
from tqdm import tqdm
import time
import copy
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.video import r3d_18 , mvit_v2_s
from torchvision.models.video.swin_transformer import swin3d_t
from torch.utils.tensorboard import SummaryWriter
from dataset import Custom3DDataset, train_transforms, depth_transforms, mvit_transform, swin_transform, \
    depth_tt_transforms, Custom4DDataset
from utils import set_seed
import json
from torchmetrics import ConfusionMatrix
from collections import defaultdict


def train_model(model, train_loader, criterion, optimizer, device, num_classes):

    print("Training...")

    model.train()
    running_loss = 0.0
    predictions = defaultdict(list)
    ground_truth = defaultdict(list)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    start_time = time.time()

    for inputs, labels, sample_ids in tqdm(train_loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        print("Input shape:", inputs.shape)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)

        # Store predictions and labels for majority voting
        for i, identifier in enumerate(sample_ids):
            predictions[identifier].append(preds[i].item())
            ground_truth[identifier].append(labels[i].item())

    # Majority voting for training accuracy
    final_preds = []
    final_labels = []
    for identifier in predictions:
        # Majority vote for each identifier
        most_common_pred = max(set(predictions[identifier]), key=predictions[identifier].count)
        most_common_label = max(set(ground_truth[identifier]), key=ground_truth[identifier].count)  # Assuming all labels per identifier are the same
        final_preds.append(most_common_pred)
        final_labels.append(most_common_label)

        # Class-wise accuracy
        if most_common_pred == most_common_label:
            class_correct[most_common_label] += 1
        class_total[most_common_label] += 1

    final_preds = torch.tensor(final_preds)
    final_labels = torch.tensor(final_labels)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = (final_preds == final_labels).float().mean()
    epoch_time = time.time() - start_time
    class_acc = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)}
    print(f"Training epoch took: {epoch_time:.2f}s")

    return epoch_loss, epoch_acc.item(), class_acc


def evaluate_model(model, val_loader, criterion, device, num_classes ):

    print("Testing...")

    model.eval()
    running_loss = 0.0
    predictions = defaultdict(list)
    ground_truth = defaultdict(list)
    confmat = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels, sample_id in tqdm(val_loader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)

            for i, ss_id in enumerate(sample_id):
                predictions[ss_id].append(preds[i].item())
                ground_truth[ss_id].append(labels[i].item())

    # Majority voting and metrics computation
    final_preds = []
    final_labels = []
    for ss_id in predictions:
        # Majority vote for each sample ID
        most_common_pred = max(set(predictions[ss_id]), key=predictions[ss_id].count)
        most_common_label = max(set(ground_truth[ss_id]),
                                key=ground_truth[ss_id].count)  # Assuming all labels per sample ID are the same
        final_preds.append(most_common_pred)
        final_labels.append(most_common_label)

    final_preds = torch.tensor(final_preds, device=device)
    final_labels = torch.tensor(final_labels, device=device)

    # Update and compute the confusion matrix
    confmat.update(final_preds, final_labels)
    confusion_matrix = confmat.compute()

    # class-wise accuracy
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for label, pred in zip(final_labels, final_preds):
        if pred == label:
            class_correct[label] += 1
        class_total[label] += 1

    class_accuracy = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)}
    accuracy = (final_preds == final_labels).float().mean()

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_time = time.time() - start_time

    print(f"Validation epoch took: {epoch_time:.2f}s")
    return epoch_loss, accuracy.item(), class_accuracy, confusion_matrix


def initialize_model(args , input_channels):
    # Load the model
    if args.backbone == "r3d":
        if args.weights:
            model = r3d_18(weights="DEFAULT")
        else:
            model = r3d_18(weights=None)

        # Modify the first convolution layer to accept a variable number of input channels
        old_conv1 = model.stem[0]
        new_conv1 = torch.nn.Conv3d(input_channels, old_conv1.out_channels,
                                    kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                                    padding=old_conv1.padding, bias=old_conv1.bias)

        if input_channels == 3:
            return model
        else:
            with torch.no_grad():
                # Initialize weights for the new channel dimensions
                torch.nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
                if new_conv1.bias is not None:
                    torch.nn.init.constant_(new_conv1.bias, 0)

            # Replace the first convolutional layer
            model.stem[0] = new_conv1
            return model
    if args.backbone == "mViT":
        if args.weights:
            model = mvit_v2_s(weights = "DEFAULT")
        else:
            model = mvit_v2_s()
        old_first_layer = model.conv_proj
        new_first_layer = nn.Conv3d(input_channels,old_first_layer.out_channels,
                                    kernel_size=old_first_layer.kernel_size,
                                    stride=old_first_layer.stride,
                                    padding=old_first_layer.padding,
                                    bias=(old_first_layer.bias is not None))

        if input_channels == 3:
            return model
        else:
            # Reinitialize the weights for the new Conv3d layer
            nn.init.kaiming_normal_(new_first_layer.weight, mode='fan_out', nonlinearity='relu')
            if new_first_layer.bias is not None:
                nn.init.constant_(new_first_layer.bias, 0)
            # Replace the original first layer with the new layer
            # print("before" , model.conv_proj)
            model.conv_proj = new_first_layer
            # print("after" ,model.conv_proj)
            return model
    if args.backbone == "swin_t":
        if args.weights:
            model = swin3d_t(weights = "DEFAULT")
        else:
            model = swin3d_t()

        old_first_layer = model.patch_embed.proj
        # Create a new Conv3d layer with the specified number of input channels but retaining other parameters
        new_first_layer = nn.Conv3d(input_channels,
                                    old_first_layer.out_channels,
                                    kernel_size=old_first_layer.kernel_size,
                                    stride=old_first_layer.stride,
                                    padding=old_first_layer.padding,
                                    bias=(old_first_layer.bias is not None))
        if input_channels == 3:
            return model
        else:
            # Reinitialize the weights for the new Conv3d layer
            nn.init.kaiming_normal_(new_first_layer.weight, mode='fan_out', nonlinearity='relu')
            if new_first_layer.bias is not None:
                nn.init.constant_(new_first_layer.bias, 0)

            # Replace the original Conv3d layer with the new one
            model.patch_embed.proj = new_first_layer
            return model




def train_and_evaluate(args):
    checkpoints_dir = os.path.join(args.base_dir, 'checkpoints')
    results_dir = os.path.join(args.base_dir, 'results')
    figures_dir = os.path.join(args.base_dir, 'figures')
    tensorboard_dir = os.path.join(args.base_dir, 'runs')

    # Create directories if they do not exist
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    set_seed(args.seed)

    if args.env == "livingroom":
        include_classes = [ 'Sleeping','Playing video game', 'Exercising', 'Using handheld smart devices', 'Reading'
            ,'Writing',  'Working on a computer', 'Watching TV', 'Carrying object']

    elif args.env == "kitchen":
        include_classes = ['Making pancake', 'Making a cup of coffee in coffee maker', 'Stocking up pantry', 'Dining',
          'Cleaning dishes',  'Using handheld smart devices',
         'Organizing the kitchen',  'Making a salad', 'Cleaning the kitchen',
         'Making a cup of instant coffee']

    elif args.env == "limited set":
        include_classes = [ 'Sleeping','Playing video game',  'Exercising', 'Using handheld smart devices', 'Reading'
            ,'Writing',  'Working on a computer', 'Watching TV', 'Carrying object' , 'Making pancake',
            'Making a cup of coffee in coffee maker', 'Stocking up pantry', 'Dining','Cleaning dishes',
            'Using handheld smart devices', 'Making a salad', 'Making a cup of instant coffee']
    else:
        include_classes = []

    # if args.backbone == 'r3d':
    #     transform = train_transforms
    #     if args.cam_view == "depth_1" or args.cam_view == "depth_2":
    #         # print(f"cam view is {args.cam_view}")
    #         transform = depth_transforms
    # if args.backbone == 'mViT':
    #     # print("hello! mViT backbone is talking...")
    #     transform = mvit_transform
    #     if args.cam_view == "depth_1" or args.cam_view == "depth_2":
    #         # print(f"cam view is {args.cam_view}")
    #         transform = depth_tt_transforms
    # if args.backbone == "swin_t":
    #     transform = swin_transform
    #     if args.cam_view == "depth_1" or args.cam_view == "depth_2":
    #         # print(f"cam view is {args.cam_view}")
    #         transform = depth_tt_transforms
    if args.cam_view == "depth_1" or args.cam_view == "depth_2":
        if args.backbone == 'r3d':
            transform = depth_transforms
        else:
            transform = depth_tt_transforms
    else:
        if args.backbone == 'r3d':
            transform = train_transforms
            depth_transform = depth_transforms
        if args.backbone == 'mViT':
            transform = mvit_transform
            depth_transform = depth_tt_transforms
        if args.backbone == "swin_t":
            transform = swin_transform
            depth_transform = depth_tt_transforms

    print(f'selected transform for {args.backbone} is {transform} \n')

    if args.modality == "rgbd":
        print(f"creating dataset for {args.modality}")
        depth_data_dir = args.data_dir.replace("rgb_dataset" , "depth_dataset")

        train_dataset = Custom4DDataset(rgb_root_dir=os.path.join(args.data_dir, 'train'),depth_root_dir =os.path.join(depth_data_dir , 'train') , cam_view= args.cam_view,
                                        transform=transform, depth_transform =depth_transform, sampling=args.sampling, include_classes=include_classes)
        val_dataset = Custom4DDataset(rgb_root_dir=os.path.join(args.data_dir, 'validation'), depth_root_dir =os.path.join(depth_data_dir , 'validation') , cam_view= args.cam_view,
                                      transform=transform,depth_transform = depth_transform, sampling=args.sampling, include_classes=include_classes)
        test_dataset = Custom4DDataset(rgb_root_dir=os.path.join(args.data_dir, 'test') ,depth_root_dir =os.path.join(depth_data_dir ,'test'), cam_view= args.cam_view ,
                                       transform=transform,depth_transform = depth_transform, sampling=args.sampling, include_classes=include_classes)
        print(f"size of trainset {len(train_dataset)} \t validationset {len(val_dataset)} \t testset {len(test_dataset)}")
    else:
        train_dataset = Custom3DDataset(root_dir=os.path.join(args.data_dir, 'train' , args.cam_view), transform=transform , sampling=args.sampling , include_classes= include_classes )
        val_dataset = Custom3DDataset(root_dir=os.path.join(args.data_dir, 'validation' , args.cam_view) , transform=transform , sampling=args.sampling , include_classes= include_classes)
        test_dataset = Custom3DDataset(root_dir=os.path.join(args.data_dir, 'test' , args.cam_view), transform=transform , sampling=args.sampling, include_classes= include_classes)

    seq_len = train_dataset.sequence_length
    classes = train_dataset.classes
    print(f'include classes:{classes}')
    sampling_method = train_dataset.sampling

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    plot_sequences(train_dataset , figures_dir)

    batch_size = train_loader.batch_size
    num_classes = len(classes)

    # Initialize the model
    if args.backbone == "r3d":
        if args.modality == "depth":
            print(f"initialized model for {args.modality} {args.backbone}")
            model = initialize_model(args,1 )
        if args.modality == "rgbd":
            print(f"initialized model for {args.modality} {args.backbone}")
            model = initialize_model(args, 4)
        #model.fc = nn.Linear(model.fc.in_features, num_classes)

        if args.modality == 'rgb':
            print(f"initialized model for {args.modality} {args.backbone}")
            if args.weights:
                print(f"Using pretrained weights")
                model = r3d_18(weights="DEFAULT")
            else:
                model = r3d_18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    if args.backbone == 'mViT':
        if args.modality == "depth":
            print(f"initialized model for {args.modality} {args.backbone}")
            model = initialize_model(args, 1)
        if args.modality == "rgbd":
            print(f"initialized model for {args.modality} {args.backbone}")
            model = initialize_model(args, 4)
        if args.modality == 'rgb':
            print(f"initialized model for {args.modality} {args.backbone}")
            if args.weights:
                print(f"Using pretrained weights")
                model = mvit_v2_s(weights='DEFAULT')
            else:
                model = mvit_v2_s(weights=None)

        if hasattr(model.head, 'in_features'):
            in_features = model.head.in_features
        else:
            # If the architecture is wrapped in a Sequential
            in_features = model.head[-1].in_features

        model.head = nn.Linear(in_features, num_classes)
    if args.backbone == 'swin_t':
        if args.modality == "depth":
            print(f"initialized model for {args.modality} {args.backbone}")
            model = initialize_model(args, 1)
        if args.modality == "rgbd":
            print(f"initialized model for {args.modality} {args.backbone}")
            model = initialize_model(args, 4)
        if args.modality == 'rgb':
            print(f"initialized model for {args.modality} {args.backbone}")
            if args.weights:
                print(f"Using pretrained weights")
                model = swin3d_t(weights = "DEFAULT")
            else:
                model = swin3d_t(weights=None)

        if hasattr(model.head, 'in_features'):
            in_features = model.head.in_features
        else:
            # If the architecture is wrapped in a Sequential
            in_features = model.head[-1].in_features

        model.head = nn.Linear(in_features, num_classes)


    # print(model)


    # Move the model to the appropriate device
    if args.device:
        device = args.device
    else:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("running on" , device)

    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.01)

    ################ Debug #########################
    # unique_labels = set()
    # for _, labels, _ in train_loader:
    #     unique_labels.update(labels.numpy())  # Assuming labels are on CPU
    #
    # print("Unique labels found:", unique_labels)
    #
    # # Check if any label is outside the expected range
    # print("Any invalid labels:", any(label < 0 or label >= num_classes for label in unique_labels))
    #
    # # Ensure the final layer of your model is appropriate
    # print("Model's final layer:", model)
    #
    # # Check the size of outputs from the model
    # for inputs, _, _ in train_loader:
    #     inputs = inputs.to(device)
    #     outputs = model(inputs)
    #     print("Output size:", outputs.size())  # Should be [batch_size, num_classes]
    #     break
    ################ Debug #########################


    writer = SummaryWriter(os.path.join(tensorboard_dir ,'activity_classification'))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stopping_patience = args.epochs // 3
    if args.modality == "rgbd":
        early_stopping_patience = args.epochs // 2
    early_stopping_counter = 0

    results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_class_acc': [],
        'val_class_acc': [],
        'test_loss': None,
        'test_acc': None,
        'test_class_acc': None
    }
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print('-' * 10)

        epoch_start_time = time.time()

        # train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device, args)
        # val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        train_loss, train_acc, train_class_acc = train_model(model, train_loader, criterion, optimizer, device, num_classes)
        val_loss, val_acc, val_class_acc , _ = evaluate_model(model, val_loader, criterion, device, num_classes)

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} took: {epoch_duration:.2f}s")

        print(f'Training Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        results['train_class_acc'].append(train_class_acc)
        results['val_class_acc'].append(val_class_acc)


        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

        scheduler.step(val_loss)

    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)


    writer.close()

    # Save the best model weights
    model_path = os.path.join(checkpoints_dir, f'best_3d_resnet_{args.modality}_{args.backbone}_{sampling_method}_seed {args.seed}_ep {args.epochs}_B {batch_size} T {seq_len}_{args.env}_{args.cam_view}_weights {args.weights}.pth')
    torch.save(model.state_dict(), model_path)
    print("checkpoint saved")

    # Evaluate on test set
    test_loss, test_acc, test_class_acc , val_conf_matrix = evaluate_model(model, test_loader, criterion, device, num_classes)

    plot_confusion_matrix(val_conf_matrix.cpu().numpy(), classes,
                          os.path.join(figures_dir,
                                       f'confusion_matrix_{args.modality}_{args.backbone}_{sampling_method}_seed {args.seed}_ep {args.epochs}_B {batch_size} T {seq_len}_{args.env}_{args.cam_view}_weights {args.weights}.png'))
    print("confusion matrix saved")
    results['test_loss'] = test_loss
    results['test_acc'] = test_acc
    results['test_class_acc'] = test_class_acc

    print(f'Test Acc: {test_acc:.4f}')

    # Save results to JSON file
    results_path = os.path.join(results_dir, f'result_{args.backbone}_{args.modality}_{sampling_method}_B {batch_size} T {seq_len}_seed {args.seed}_ep {args.epochs}_{args.env}_{args.cam_view}_weights {args.weights}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    print("result saved")
    # Plot and save the training and validation loss and accuracy
    epochs = range(1, len(train_losses) + 1)
    savepath = os.path.join(figures_dir ,f'training_validation_loss_accuracy {args.modality}_{args.backbone}_{sampling_method}_B {batch_size} T {seq_len}_seed {args.seed}_ep {args.epochs}_{args.env}_{args.cam_view}_weights {args.weights}.png')
    plot_training_validation_loss_accuracy(epochs, train_losses, val_losses, train_accuracies, val_accuracies,
                                           args.seed, savepath)
    print("accuracy plot saved")

    return results

def check_class_distribution(dataset):
    from collections import Counter
    class_counts = Counter()
    for _, label, _ in dataset:
        class_counts[label] += 1
    return class_counts

if "__name__ == __main__":
    print("Train and evaluate module as main.")
    # main_parser = argparse.ArgumentParser(description='Train and evaluate a 3D ResNet model in main.')
    # main_args = main_parser.parse_args()
    # main_args.seed = 1
    # main_args.env = "limited set"
    # main_args.epochs = 2
    # main_args.base_dir = "/home/ghazal/Activity_Recognition_benchmarking/"
    # main_args.batch_size = 8
    # main_args.backbone = "mvit"
    # main_args.sampling = "single-random"
    # main_args.device = "cuda:1"
    # main_args.cam_view = "cam_2"
    # main_args.data_dir ="/mnt/data-tmp/ghazal/DARai_DATA/rgb_dataset/"
    # #
    # include_classes = ['Sleeping', 'Playing video game', 'Exercising', 'Using handheld smart devices', 'Reading'
    #     , 'Writing', 'Working on a computer', 'Watching TV', 'Carrying object', 'Making pancake',
    #                    'Making a cup of coffee in coffee maker', 'Stocking up pantry', 'Dining', 'Cleaning dishes',
    #                    'Using handheld smart devices', 'Making a salad', 'Making a cup of instant coffee']
    #
    # train_dataset = Custom3DDataset(root_dir=os.path.join(main_args.data_dir, 'train', main_args.cam_view),
    #                                 transform=mvit_transform, sampling=main_args.sampling, include_classes=include_classes)
    # val_dataset = Custom3DDataset(root_dir=os.path.join(main_args.data_dir, 'validation', main_args.cam_view),
    #                               transform=train_transforms, sampling=main_args.sampling, include_classes=include_classes)
    # test_dataset = Custom3DDataset(root_dir=os.path.join(main_args.data_dir, 'test', main_args.cam_view),
    #                                transform=test_transforms, sampling=main_args.sampling, include_classes=include_classes)
    # train_loader = DataLoader(train_dataset, batch_size=main_args.batch_size, shuffle=True)
    #
    # for x , _ , _ in train_loader:
    #     print(x.shape)
    #     break
    # train_class_distribution = check_class_distribution(train_dataset)
    # print("Training class distribution cam_2:", train_class_distribution)
    # val_class_distribution = check_class_distribution(val_dataset)
    # print("Validation class distribution cam_2:", val_class_distribution)
    # test_class_distribution = check_class_distribution(test_dataset)
    # print("Test class distribution cam_2:", test_class_distribution)
    #

    # train_and_evaluate(main_args)
