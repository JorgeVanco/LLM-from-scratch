#!/usr/bin/env python3
"""
Results analyzer for hyperparameter search
Usage: python analyze_results.py <results_csv>
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import argparse
from pathlib import Path

def load_and_validate_data(csv_file):
    """Load and validate the results CSV file."""
    try:
        df = pd.read_csv(csv_file)
        required_columns = ['learning_rate', 'final_loss', 'best_val_loss', 'status']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

def create_analysis_plots(df, output_dir="."):
    """Create comprehensive analysis plots."""
    # Filter completed experiments
    completed_df = df[df['status'] == 'completed'].copy()
    
    if completed_df.empty:
        print("No completed experiments found")
        return
    
    # Convert learning rates to numeric for proper sorting
    completed_df['learning_rate'] = pd.to_numeric(completed_df['learning_rate'])
    completed_df['final_loss'] = pd.to_numeric(completed_df['final_loss'], errors='coerce')
    completed_df['best_val_loss'] = pd.to_numeric(completed_df['best_val_loss'], errors='coerce')
    
    # Sort by learning rate
    completed_df = completed_df.sort_values('learning_rate')
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # Create main analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hyperparameter Search Results Analysis', fontsize=16)
    
    # Plot 1: Learning rate vs validation loss
    ax1 = axes[0, 0]
    ax1.semilogx(completed_df['learning_rate'], completed_df['best_val_loss'], 'bo-', markersize=8)
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Best Validation Loss')
    ax1.set_title('Learning Rate vs Validation Loss')
    ax1.grid(True, alpha=0.3)
    
    # Highlight best result
    best_idx = completed_df['best_val_loss'].idxmin()
    best_lr = completed_df.loc[best_idx, 'learning_rate']
    best_val_loss = completed_df.loc[best_idx, 'best_val_loss']
    ax1.plot(best_lr, best_val_loss, 'ro', markersize=12, label=f'Best: LR={best_lr:.0e}')
    ax1.legend()
    
    # Plot 2: Learning rate vs final loss
    ax2 = axes[0, 1]
    ax2.semilogx(completed_df['learning_rate'], completed_df['final_loss'], 'go-', markersize=8)
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Final Training Loss')
    ax2.set_title('Learning Rate vs Final Training Loss')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss comparison
    ax3 = axes[1, 0]
    x = np.arange(len(completed_df))
    width = 0.35
    
    ax3.bar(x - width/2, completed_df['final_loss'], width, label='Final Training Loss', alpha=0.7)
    ax3.bar(x + width/2, completed_df['best_val_loss'], width, label='Best Validation Loss', alpha=0.7)
    
    ax3.set_xlabel('Experiment Index')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training vs Validation Loss Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{lr:.0e}' for lr in completed_df['learning_rate']], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Duration analysis (if available)
    ax4 = axes[1, 1]
    if 'duration_minutes' in completed_df.columns:
        completed_df['duration_minutes'] = pd.to_numeric(completed_df['duration_minutes'], errors='coerce')
        ax4.semilogx(completed_df['learning_rate'], completed_df['duration_minutes'], 'mo-', markersize=8)
        ax4.set_xlabel('Learning Rate')
        ax4.set_ylabel('Training Duration (minutes)')
        ax4.set_title('Learning Rate vs Training Duration')
        ax4.grid(True, alpha=0.3)
    else:
        # Alternative: Show loss distribution
        ax4.hist(completed_df['best_val_loss'], bins=10, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Best Validation Loss')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Best Validation Loss')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'lr_search_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved as '{output_path}'")
    plt.close()
    
    # Create summary statistics plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot of validation losses
    ax.boxplot(completed_df['best_val_loss'], labels=['Validation Loss'])
    ax.set_ylabel('Loss')
    ax.set_title('Distribution of Best Validation Losses')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""
    Statistics:
    Mean: {completed_df['best_val_loss'].mean():.4f}
    Median: {completed_df['best_val_loss'].median():.4f}
    Std: {completed_df['best_val_loss'].std():.4f}
    Min: {completed_df['best_val_loss'].min():.4f}
    Max: {completed_df['best_val_loss'].max():.4f}
    """
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    summary_path = Path(output_dir) / 'lr_search_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Summary plot saved as '{summary_path}'")
    plt.close()

def print_detailed_analysis(df):
    """Print detailed analysis of the results."""
    completed_df = df[df['status'] == 'completed'].copy()
    failed_df = df[df['status'] == 'failed'].copy()
    
    print("\n" + "="*50)
    print("DETAILED ANALYSIS")
    print("="*50)
    
    print(f"Total experiments: {len(df)}")
    print(f"Completed: {len(completed_df)}")
    print(f"Failed: {len(failed_df)}")
    
    if completed_df.empty:
        print("No completed experiments to analyze")
        return
    
    # Convert to numeric
    completed_df['learning_rate'] = pd.to_numeric(completed_df['learning_rate'])
    completed_df['final_loss'] = pd.to_numeric(completed_df['final_loss'], errors='coerce')
    completed_df['best_val_loss'] = pd.to_numeric(completed_df['best_val_loss'], errors='coerce')
    
    # Best results
    best_val_idx = completed_df['best_val_loss'].idxmin()
    best_final_idx = completed_df['final_loss'].idxmin()
    
    print(f"\nBest Validation Loss:")
    print(f"  Learning Rate: {completed_df.loc[best_val_idx, 'learning_rate']:.0e}")
    print(f"  Validation Loss: {completed_df.loc[best_val_idx, 'best_val_loss']:.4f}")
    print(f"  Final Training Loss: {completed_df.loc[best_val_idx, 'final_loss']:.4f}")
    
    print(f"\nBest Final Training Loss:")
    print(f"  Learning Rate: {completed_df.loc[best_final_idx, 'learning_rate']:.0e}")
    print(f"  Final Training Loss: {completed_df.loc[best_final_idx, 'final_loss']:.4f}")
    print(f"  Validation Loss: {completed_df.loc[best_final_idx, 'best_val_loss']:.4f}")
    
    # Statistics
    print(f"\nValidation Loss Statistics:")
    print(f"  Mean: {completed_df['best_val_loss'].mean():.4f}")
    print(f"  Median: {completed_df['best_val_loss'].median():.4f}")
    print(f"  Std: {completed_df['best_val_loss'].std():.4f}")
    print(f"  Min: {completed_df['best_val_loss'].min():.4f}")
    print(f"  Max: {completed_df['best_val_loss'].max():.4f}")
    
    # Learning rate analysis
    print(f"\nLearning Rate Analysis:")
    lr_sorted = completed_df.sort_values('best_val_loss')
    print(f"  Top 3 Learning Rates (by validation loss):")
    for i, (_, row) in enumerate(lr_sorted.head(3).iterrows()):
        print(f"    {i+1}. LR: {row['learning_rate']:.0e}, Val Loss: {row['best_val_loss']:.4f}")
    
    # Failed experiments
    if not failed_df.empty:
        print(f"\nFailed Experiments:")
        failed_df['learning_rate'] = pd.to_numeric(failed_df['learning_rate'])
        for _, row in failed_df.iterrows():
            print(f"  Learning Rate: {row['learning_rate']:.0e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter search results')
    parser.add_argument('csv_file', help='Path to the results CSV file')
    parser.add_argument('--output-dir', default='.', help='Output directory for plots')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Load data
    df = load_and_validate_data(args.csv_file)
    
    # Print detailed analysis
    print_detailed_analysis(df)
    
    # Create plots
    if not args.no_plots:
        try:
            create_analysis_plots(df, args.output_dir)
        except ImportError as e:
            print(f"Warning: Could not create plots due to missing dependencies: {e}")
            print("Install matplotlib and seaborn to enable plotting")
        except Exception as e:
            print(f"Error creating plots: {e}")

if __name__ == "__main__":
    main()