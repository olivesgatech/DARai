import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import torch
from torch.utils.data import DataLoader , random_split
from tqdm import tqdm
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.video import r3d_18 , mvit_v2_s
from torchvision.models.video.swin_transformer import swin3d_t
from torch.utils.tensorboard import SummaryWriter
from dataset import Custom3DDataset , train_transforms ,depth_transforms , mvit_transform , swin_transform, depth_tt_transforms
from utils import set_seed
import json
from torchmetrics import ConfusionMatrix
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-view Test for Activity Recognition")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory for data (should contain test/<cam_view> folder)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the saved model checkpoint (.pth file)')
    parser.add_argument("--device", type=str, required=False)
    parser.add_argument('--cam_view', type=str, required=True,
                        help='Name of the camera view folder to use for testing (e.g., cam2)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for DataLoader')
    parser.add_argument('--sampling', type=str, default="multi-uniform",
                        help='Sampling method to use for dataset loading')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON filename (if not provided, defaults to cross_view_<cam_view>.json)')
    parser.add_argument('--backbone' , type = str , default= 'r3d' , help = "backbone of the model")
    parser.add_argument("--modality" , type= str , default= 'rgb')
    parser.add_argument("--env" , type = str , default= "both" , help = "set of classes to train and test on")
    parser.add_argument("--pretrained" , type = str , default="False" , help = "pretrained model False or True")
    return parser.parse_args()

def initialize_model(args , input_channels):
    # Load the model
    if args.backbone == "r3d":

        model = r3d_18(weights=None)

        # Modify the first convolution layer to accept a variable number of input channels
        old_conv1 = model.stem[0]
        new_conv1 = torch.nn.Conv3d(input_channels, old_conv1.out_channels,
                                    kernel_size=old_conv1.kernel_size, stride=old_conv1.stride,
                                    padding=old_conv1.padding, bias=old_conv1.bias)
        with torch.no_grad():
            # Initialize weights for the new channel dimensions
            torch.nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
            if new_conv1.bias is not None:
                torch.nn.init.constant_(new_conv1.bias, 0)
        if input_channels == 3:
            return model
        else:
            # Replace the first convolutional layer
            model.stem[0] = new_conv1
            print(f"return {args.backbone} model with input channel {input_channels}")
            return model
    if args.backbone == "mViT":
        model = mvit_v2_s()
        old_first_layer = model.conv_proj
        new_first_layer = nn.Conv3d(in_channels = input_channels,out_channels = old_first_layer.out_channels,
                                    kernel_size=old_first_layer.kernel_size,
                                    stride=old_first_layer.stride,
                                    padding=old_first_layer.padding,
                                    bias=(old_first_layer.bias is not None))
        # Reinitialize the weights for the new Conv3d layer
        nn.init.kaiming_normal_(new_first_layer.weight, mode='fan_out', nonlinearity='relu')
        if new_first_layer.bias is not None:
            nn.init.constant_(new_first_layer.bias, 0)

        if input_channels == 3:
            return model
        else:
            # Replace the original first layer with the new layer
            # print("before" , model.conv_proj)
            model.conv_proj = new_first_layer
            # print("after" ,model.conv_proj)
            # print(f"return {args.backbone} model with input channel {input_channels}")
            return model
    if args.backbone == "swin_t":

        model = swin3d_t()

        old_first_layer = model.patch_embed.proj
        # Create a new Conv3d layer with the specified number of input channels but retaining other parameters
        new_first_layer = nn.Conv3d(input_channels,
                                    old_first_layer.out_channels,
                                    kernel_size=old_first_layer.kernel_size,
                                    stride=old_first_layer.stride,
                                    padding=old_first_layer.padding,
                                    bias=(old_first_layer.bias is not None))
        # Reinitialize the weights for the new Conv3d layer
        nn.init.kaiming_normal_(new_first_layer.weight, mode='fan_out', nonlinearity='relu')
        if new_first_layer.bias is not None:
            nn.init.constant_(new_first_layer.bias, 0)

        if input_channels == 3:
            return model
        else:
            # Replace the original Conv3d layer with the new one
            model.patch_embed.proj = new_first_layer
            print(f"return {args.backbone} model with input channel {input_channels}")
            return model


def main():
    args = parse_args()
    set_seed(1000)

    # If no output filename is given, default to "cross_view_<cam_view>.json"
    output_filename = os.path.join(args.output ,f"cross_view_{args.env}_{args.backbone}_pretraining {args.pretrained}_test on {args.cam_view}.json")

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

    if args.backbone == 'r3d':
        transform = train_transforms
        if args.cam_view == "depth_1" or args.cam_view == "depth_2":
            # print(f"cam view is {args.cam_view}")
            transform = depth_transforms
    if args.backbone == 'mViT':
        # print("hello! mViT backbone is talking...")
        transform = mvit_transform
        if args.cam_view == "depth_1" or args.cam_view == "depth_2":
            # print(f"cam view is {args.cam_view}")
            transform = depth_tt_transforms
    if args.backbone == "swin_t":
        transform = swin_transform
        if args.cam_view == "depth_1" or args.cam_view == "depth_2":
            # print(f"cam view is {args.cam_view}")
            transform = depth_tt_transforms

    test_root = os.path.join(args.data_dir, 'test', args.cam_view)
    print(f"Testing on {test_root} with {args.checkpoint_path}")
    test_dataset = Custom3DDataset(
        root_dir=test_root,
        transform=transform,
        sampling="multi-uniform", include_classes= include_classes)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test loader {len(test_loader)}")
    num_classes = len(test_dataset.classes)

    # if args.backbone == "r3d":
    #     model = r3d_18(weights=None)
    #     model.fc = nn.Linear(model.fc.in_features, num_classes)
    # if args.backbone == 'mViT':
    #     model = mvit_v2_s(weights=None)
    #     if hasattr(model.head, 'in_features'):
    #         in_features = model.head.in_features
    #     else:
    #         # If the architecture is wrapped in a Sequential
    #         in_features = model.head[-1].in_features
    #     model.head = nn.Linear(in_features, num_classes)
    # if args.backbone == 'swin_t':
    #     model = swin3d_t(weights=None)
    #     if hasattr(model.head, 'in_features'):
    #         in_features = model.head.in_features
    #     else:
    #         # If the architecture is wrapped in a Sequential
    #         in_features = model.head[-1].in_features
    #
    #     model.head = nn.Linear(in_features, num_classes)
    if args.backbone == "r3d":
        if args.modality == "depth":
            model = initialize_model(args,1)
        if args.modality == "rgbd":
            model = initialize_model(args, 4)

        if args.modality == 'rgb':
            model = r3d_18(weights=None)


        model.fc = nn.Linear(model.fc.in_features, num_classes)

    if args.backbone == 'mViT':
        if args.modality == "depth":
            model = initialize_model(args, 1)
            #print(f"initialized {args.backbone} with {args.modality} with input channel 1")
        if args.modality == "rgbd":
            model = initialize_model(args, 4)
        if args.modality == 'rgb':
            model = mvit_v2_s(weights = None)

        if hasattr(model.head, 'in_features'):
            in_features = model.head.in_features
        else:
            # If the architecture is wrapped in a Sequential
            in_features = model.head[-1].in_features
        # print(f'number of classes when initializing the model {num_classes}')

        model.head = nn.Linear(in_features, num_classes)

    if args.backbone == 'swin_t':
        if args.modality == "depth":
            model = initialize_model(args, 1)
            #print(f"initialized {args.backbone} with {args.modality} with input channel 1")
        if args.modality == "rgbd":
            model = initialize_model(args, 4)

        if args.modality == 'rgb':
            model = swin3d_t(weights = None)

        if hasattr(model.head, 'in_features'):
            in_features = model.head.in_features
        else:
            # If the architecture is wrapped in a Sequential
            in_features = model.head[-1].in_features
        model.head = nn.Linear(in_features, num_classes)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()  # set model to evaluation mode

    # Move model to device (GPU if available)
    device = args.device
    print(f"Using {device}")
    model.to(device)


    total_samples = 0
    correct_predictions = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        # for inputs, labels in test_loader:
        for inputs, labels, _ in tqdm(test_loader, desc="Inference", total=len(test_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)  # output shape: [batch_size, num_classes]
            _, preds = torch.max(outputs, 1)

            total_samples += labels.size(0)
            correct_predictions += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    overall_accuracy = correct_predictions / total_samples * 100
    print(f"Overall Test Accuracy on alternate camera data: {overall_accuracy:.2f}%")

    y_true = torch.tensor(all_labels)
    y_pred = torch.tensor(all_preds)

    # cm = ConfusionMatrix(y_pred, y_true, num_classes=num_classes)
    cm_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    cm_metric.update(y_pred, y_true)
    cm = cm_metric.compute()
    cm_list = cm.cpu().tolist()  # convert tensor to list for JSON serialization

    # Compute per-class accuracy from the confusion matrix.
    per_class_accuracy = {}
    for i in range(num_classes):
        # Sum of the i-th row corresponds to the total true samples for class i.
        total_class_samples = sum(cm_list[i])
        if total_class_samples > 0:
            acc = (cm_list[i][i] / total_class_samples) * 100
        else:
            acc = 0.0

        class_name = test_dataset.classes[i] if hasattr(test_dataset, 'classes') else f"class_{i}"
        per_class_accuracy[class_name] = acc


    # Save the results to a JSON file

    results = {
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": cm_list
    }

    with open(output_filename, 'w') as fp:
        json.dump(results, fp, indent=4)

    print(f"Results saved to {output_filename}")


if __name__ == '__main__':
    main()


#Example Usage
#python "./3D_sequence_model/cross-view inference.py" --data_dir --checkpoint_path --cam_view --batch_size --output