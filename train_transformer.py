import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import time
from pathlib import Path
import numpy as np

# Import our custom modules
from core.data.dataset import PlantDataset
from core.models.transformer import PlantIdentificationModel, VisionTransformer


def get_transforms(img_size=224):
    """Get transforms for training and validation/testing."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return train_transform, val_transform


# Define utility functions
class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    """Save checkpoint to disk."""
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        import shutil
        shutil.copyfile(filepath, os.path.join(output_dir, 'model_best.pth'))


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, device, print_freq=10):
    """Train for one epoch."""
    model.train()
    
    # Create metrics trackers
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # Start timing
    end = time.time()
    
    # Iterate over batches
    for i, (images, targets) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move data to device
        images = images.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Compute gradients and update parameters
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update learning rate if using cosine scheduler
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        
        # Compute accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, min(5, len(train_loader.dataset.classes))))
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print progress
        if i % print_freq == 0:
            print(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'
            )
    
    return losses.avg, top1.avg, top5.avg


def validate(model, val_loader, criterion, device, print_freq=10):
    """Validate the model."""
    model.eval()
    
    # Create metrics trackers
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # Start timing
    end = time.time()
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Compute accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, min(5, len(val_loader.dataset.classes))))
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Print progress
            if i % print_freq == 0:
                print(
                    f'Test: [{i}/{len(val_loader)}] '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                    f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                    f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'
                )
    
    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    
    return losses.avg, top1.avg, top5.avg


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train plant identification transformer model')
    
    # Dataset arguments
    parser.add_argument('--data-dir', type=str, default='data/plantnet_300K',
                        help='Directory containing dataset')
    parser.add_argument('--metadata-file', type=str, default='data/plantnet_300K/plantnet300K_metadata.json',
                        help='Path to dataset metadata file')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='vit_base_patch16_224',
                        help='Model type (vit_base_patch16_224, vit_small_patch16_224, custom_vit)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch size for ViT')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        help='LR scheduler (cosine or plateau)')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='output/transformer',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='Save frequency')
    
    return parser.parse_args()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training and validation curves."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Acc')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def setup_cuda():
    """Setup CUDA and optimize for performance"""
    # Force CUDA device if available
    if torch.cuda.is_available():
        # Select GPU device
        device = torch.device('cuda:0')
        
        # Print CUDA information
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Optimize CuDNN for performance
        cudnn.benchmark = True
        
        # Set memory allocation parameters to avoid fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # For mixed precision (faster training if supported)
        try:
            # Check if Ampere architecture (RTX 30xx series or newer)
            if torch.cuda.get_device_capability()[0] >= 8:
                print("Mixed precision training supported and enabled")
                # torch.set_float32_matmul_precision('high')  # Uncomment for PyTorch 2.0+
            else:
                print("Mixed precision training not optimized for this GPU")
        except:
            print("Couldn't determine GPU architecture for mixed precision setup")
    else:
        print("WARNING: CUDA not available, using CPU. Training will be very slow!")
        device = torch.device('cpu')
    
    return device


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup CUDA
    device = setup_cuda()
    print(f"Using device: {device}")
    
    # Get data transforms
    train_transform, val_transform = get_transforms(args.img_size)
    
    # Create datasets
    try:
        print(f"Loading datasets from {args.data_dir}")
        train_dataset = PlantDataset(
            data_dir=args.data_dir,
            split='images_train',
            transform=train_transform,
            metadata_file=args.metadata_file
        )
        
        val_dataset = PlantDataset(
            data_dir=args.data_dir,
            split='images_val',
            transform=val_transform,
            metadata_file=args.metadata_file
        )
        
        test_dataset = PlantDataset(
            data_dir=args.data_dir,
            split='images_test',
            transform=val_transform,
            metadata_file=args.metadata_file
        )
        
        print(f"Loaded {len(train_dataset)} training samples")
        print(f"Loaded {len(val_dataset)} validation samples")
        print(f"Loaded {len(test_dataset)} test samples")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save class mapping
    class_mapping_path = os.path.join(args.output_dir, 'class_mapping.json')
    with open(class_mapping_path, 'w') as f:
        json.dump(train_dataset.class_to_idx, f, indent=2)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # For data loading performance, optimize GPU data transfer
    if torch.cuda.is_available():
        # Pin memory for faster data transfer to GPU
        for loader in [train_loader, val_loader, test_loader]:
            loader.pin_memory = True
            
        # Set number of workers based on CPU cores, but not more than 4 for stability
        num_workers = min(4, os.cpu_count() or 1)
        for loader in [train_loader, val_loader, test_loader]:
            loader.num_workers = num_workers
        
        print(f"Data loaders configured with {num_workers} workers and pin_memory=True")
    
    # Create model
    num_classes = len(train_dataset.classes)
    print(f"Creating model: {args.model_type} with {num_classes} classes")
    
    if args.model_type == 'custom_vit':
        # Use our custom Vision Transformer implementation
        model = VisionTransformer(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout_rate=0.1,
            attn_dropout_rate=0.0
        )
    else:
        # Use timm's Vision Transformer through our wrapper
        try:
            import timm
            model = timm.create_model(
                args.model_type,
                pretrained=False,
                num_classes=num_classes
            )
        except Exception as e:
            print(f"Error creating timm model: {e}")
            print("Falling back to custom Vision Transformer")
            model = VisionTransformer(
                img_size=args.img_size,
                patch_size=args.patch_size,
                in_channels=3,
                num_classes=num_classes,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                dropout_rate=0.1,
                attn_dropout_rate=0.0
            )
    
    # Move model to device
    model = model.to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )
    
    # Define scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * len(train_loader),
            eta_min=args.lr / 100
        )
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
    else:
        scheduler = None
    
    # Initialize training variables
    best_acc1 = 0
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            
            # Map model to CPU if no GPU available
            if not torch.cuda.is_available():
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                checkpoint = torch.load(args.resume)
            
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            
            # Load model weights
            model.load_state_dict(checkpoint['state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load scheduler state if available
            if 'scheduler' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Evaluate only if requested
    if args.evaluate:
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device, args.print_freq)
        return
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch, device, args.print_freq
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device, args.print_freq)
        
        # Update learning rate scheduler if using plateau scheduler
        if args.scheduler == 'plateau' and scheduler is not None:
            scheduler.step(val_loss)
        
        # Remember best accuracy and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        
        # Save training history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc1)
        val_accs.append(val_acc1)
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            }, is_best, args.output_dir)
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Plot training curves
    try:
        plot_path = os.path.join(args.output_dir, 'training_curves.png')
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=plot_path)
    except Exception as e:
        print(f"Error plotting training curves: {e}")
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_loss, test_acc1, test_acc5 = validate(model, test_loader, criterion, device, args.print_freq)
    
    # Save final results
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy@1: {test_acc1:.2f}\n")
        f.write(f"Test Accuracy@5: {test_acc5:.2f}\n")
    
    # Save final model weights
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
    
    print("Training complete!")


if __name__ == '__main__':
    main()