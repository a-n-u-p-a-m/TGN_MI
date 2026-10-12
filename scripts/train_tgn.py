"""Train TGN on temporal motif completion task.
Usage: python -m scripts.train_tgn [--seed 42] [--output_dir runs/]

Run from the Project3 directory:
    cd Project3
    python -m scripts.train_tgn
"""
import argparse
import os
import sys
import torch

# Ensure the Project3 directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_gnn_mi.config import DatasetConfig, ModelConfig, TrainingConfig
from temporal_gnn_mi.datasets.temporal_motif_generator import TemporalMotifDataset
from temporal_gnn_mi.datasets.utils import set_seed, compute_dataset_statistics
from temporal_gnn_mi.models.tgn import TGN
from temporal_gnn_mi.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train TGN on temporal motif completion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="runs/tgn_mi", help="Output directory")
    parser.add_argument("--train_samples", type=int, default=20000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=3000, help="Number of validation samples")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Configs
    dataset_config = DatasetConfig(seed=args.seed)
    model_config = ModelConfig()
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    print("=" * 60)
    print("Temporal GNN Mechanistic Interpretability")
    print("Experiment: TGN on Temporal Motif Completion")
    print("=" * 60)
    print(f"Device: {train_config.device}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Generate datasets
    print("Generating datasets...")
    train_dataset = TemporalMotifDataset(
        config=dataset_config, 
        num_samples=args.train_samples, 
        split='train'
    )
    val_dataset = TemporalMotifDataset(
        config=dataset_config, 
        num_samples=args.val_samples, 
        split='val'
    )
    
    # Print dataset statistics
    stats = compute_dataset_statistics(train_dataset)
    print(f"Train: {stats['num_samples']} samples, "
          f"positive ratio: {stats['positive_ratio']:.2f}")
    print(f"Avg events per sample: {stats['avg_events']:.1f}")
    print()
    
    # Create TGN model
    model = TGN(
        config=model_config, 
        num_nodes=dataset_config.num_nodes,
        edge_feat_dim=16
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"TGN model: {param_count:,} parameters")
    print()
    
    # Train
    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        output_dir=args.output_dir
    )
    
    best_metrics = trainer.train()
    print()
    print("=" * 60)
    print("Training Complete!")
    print(f"Best Val ROC-AUC: {best_metrics.get('roc_auc', 'N/A')}")
    print(f"Best Val Accuracy: {best_metrics.get('accuracy', 'N/A')}")
    print(f"Best Val AP: {best_metrics.get('avg_precision', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
