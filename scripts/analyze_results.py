#!/usr/bin/env python3
"""
Generic results analyzer for hyperparameter search
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
        
        # Get the parameter name from the first column
        parameter_name = df.columns[0]
        
        # Required columns (first one is dynamic, rest are fixed)
        required_columns = [parameter_name, 'final_loss', 'best_val_loss', 'status']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        return df, parameter_name
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

def is_log_scale_appropriate(values):
    """Determine if log scale is appropriate for the given values."""
    # Check if values span multiple orders of magnitude
    if len(values) < 2:
        return False
    
    # Remove zeros and negative values for log scale consideration
    positive_values = [v for v in values if v > 0]
    if len(positive_values) < 2:
        return False
    
    min_val = min(positive_values)
    max_val = max(positive_values)
    
    # Use log scale if the range spans more than 2 orders of magnitude
    return max_val / min_val > 100

def create_analysis_plots(df, parameter_name, output_dir="."):
    """Create comprehensive analysis plots."""
    # Filter completed experiments
    completed_df = df[df['status'] == 'completed'].copy()
    
    if completed_df.empty:
        print("No completed experiments found")
        return
    
    # Convert parameter values to numeric for proper sorting
    completed_df[parameter_name] = pd.to_numeric(completed_df[parameter_name], errors='coerce')
    completed_df['final_loss'] = pd.to_numeric(completed_df['final_loss'], errors='coerce')
    completed_df['best_val_loss'] = pd.to_numeric(completed_df['best_val_loss'], errors='coerce')
    
    # Remove rows with NaN values
    completed_df = completed_df.dropna(subset=[parameter_name, 'final_loss', 'best_val_loss'])
    
    if completed_df.empty:
        print("No valid completed experiments found after data cleaning")
        return
    
    # Sort by parameter value
    completed_df = completed_df.sort_values(parameter_name)
    
    # Determine if log scale is appropriate
    use_log_scale = is_log_scale_appropriate(completed_df[parameter_name])
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # Create main analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{parameter_name.replace("_", " ").title()} Search Results Analysis', fontsize=16)
    
    # Plot 1: Parameter vs validation loss
    ax1 = axes[0, 0]
    if use_log_scale:
        ax1.semilogx(completed_df[parameter_name], completed_df['best_val_loss'], 'bo-', markersize=8)
    else:
        ax1.plot(completed_df[parameter_name], completed_df['best_val_loss'], 'bo-', markersize=8)
    ax1.set_xlabel(parameter_name.replace('_', ' ').title())
    ax1.set_ylabel('Best Validation Loss')
    ax1.set_title(f'{parameter_name.replace("_", " ").title()} vs Validation Loss')
    ax1.grid(True, alpha=0.3)
    
    # Highlight best result
    best_idx = completed_df['best_val_loss'].idxmin()
    best_param = completed_df.loc[best_idx, parameter_name]
    best_val_loss = completed_df.loc[best_idx, 'best_val_loss']
    ax1.plot(best_param, best_val_loss, 'ro', markersize=12, 
             label=f'Best: {parameter_name}={best_param:.3g}')
    ax1.legend()
    
    # Plot 2: Parameter vs final loss
    ax2 = axes[0, 1]
    if use_log_scale:
        ax2.semilogx(completed_df[parameter_name], completed_df['final_loss'], 'go-', markersize=8)
    else:
        ax2.plot(completed_df[parameter_name], completed_df['final_loss'], 'go-', markersize=8)
    ax2.set_xlabel(parameter_name.replace('_', ' ').title())
    ax2.set_ylabel('Final Training Loss')
    ax2.set_title(f'{parameter_name.replace("_", " ").title()} vs Final Training Loss')
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
    
    # Format parameter values for x-axis labels
    param_labels = []
    for param_val in completed_df[parameter_name]:
        if abs(param_val) < 0.001 or abs(param_val) >= 1000:
            param_labels.append(f'{param_val:.0e}')
        else:
            param_labels.append(f'{param_val:.3g}')
    
    ax3.set_xticklabels(param_labels, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Duration analysis (if available)
    ax4 = axes[1, 1]
    if 'duration_minutes' in completed_df.columns:
        completed_df['duration_minutes'] = pd.to_numeric(completed_df['duration_minutes'], errors='coerce')
        if use_log_scale:
            ax4.semilogx(completed_df[parameter_name], completed_df['duration_minutes'], 'mo-', markersize=8)
        else:
            ax4.plot(completed_df[parameter_name], completed_df['duration_minutes'], 'mo-', markersize=8)
        ax4.set_xlabel(parameter_name.replace('_', ' ').title())
        ax4.set_ylabel('Training Duration (minutes)')
        ax4.set_title(f'{parameter_name.replace("_", " ").title()} vs Training Duration')
        ax4.grid(True, alpha=0.3)
    else:
        # Alternative: Show loss distribution
        ax4.hist(completed_df['best_val_loss'], bins=min(10, len(completed_df)), alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Best Validation Loss')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Best Validation Loss')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / f'{parameter_name}_search_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved as '{output_path}'")
    plt.close()
    
    # Create summary statistics plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot of validation losses
    ax.boxplot(completed_df['best_val_loss'], tick_labels=['Validation Loss'])
    ax.set_ylabel('Loss')
    ax.set_title(f'Distribution of Best Validation Losses - {parameter_name.replace("_", " ").title()} Search')
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
    summary_path = Path(output_dir) / f'{parameter_name}_search_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Summary plot saved as '{summary_path}'")
    plt.close()

def print_detailed_analysis(df, parameter_name):
    """Print detailed analysis of the results."""
    completed_df = df[df['status'] == 'completed'].copy()
    failed_df = df[df['status'] == 'failed'].copy()
    
    print("\n" + "="*50)
    print("DETAILED ANALYSIS")
    print("="*50)
    
    print(f"Parameter: {parameter_name}")
    print(f"Total experiments: {len(df)}")
    print(f"Completed: {len(completed_df)}")
    print(f"Failed: {len(failed_df)}")
    
    if completed_df.empty:
        print("No completed experiments to analyze")
        return
    
    # Convert to numeric
    completed_df[parameter_name] = pd.to_numeric(completed_df[parameter_name], errors='coerce')
    completed_df['final_loss'] = pd.to_numeric(completed_df['final_loss'], errors='coerce')
    completed_df['best_val_loss'] = pd.to_numeric(completed_df['best_val_loss'], errors='coerce')
    
    # Remove rows with NaN values
    completed_df = completed_df.dropna(subset=[parameter_name, 'final_loss', 'best_val_loss'])
    
    if completed_df.empty:
        print("No valid completed experiments found after data cleaning")
        return
    
    # Best results
    best_val_idx = completed_df['best_val_loss'].idxmin()
    best_final_idx = completed_df['final_loss'].idxmin()
    
    print(f"\nBest Validation Loss:")
    print(f"  {parameter_name}: {completed_df.loc[best_val_idx, parameter_name]:.3g}")
    print(f"  Validation Loss: {completed_df.loc[best_val_idx, 'best_val_loss']:.4f}")
    print(f"  Final Training Loss: {completed_df.loc[best_val_idx, 'final_loss']:.4f}")
    
    print(f"\nBest Final Training Loss:")
    print(f"  {parameter_name}: {completed_df.loc[best_final_idx, parameter_name]:.3g}")
    print(f"  Final Training Loss: {completed_df.loc[best_final_idx, 'final_loss']:.4f}")
    print(f"  Validation Loss: {completed_df.loc[best_final_idx, 'best_val_loss']:.4f}")
    
    # Statistics
    print(f"\nValidation Loss Statistics:")
    print(f"  Mean: {completed_df['best_val_loss'].mean():.4f}")
    print(f"  Median: {completed_df['best_val_loss'].median():.4f}")
    print(f"  Std: {completed_df['best_val_loss'].std():.4f}")
    print(f"  Min: {completed_df['best_val_loss'].min():.4f}")
    print(f"  Max: {completed_df['best_val_loss'].max():.4f}")
    
    # Parameter value statistics
    print(f"\n{parameter_name.replace('_', ' ').title()} Statistics:")
    print(f"  Mean: {completed_df[parameter_name].mean():.3g}")
    print(f"  Median: {completed_df[parameter_name].median():.3g}")
    print(f"  Std: {completed_df[parameter_name].std():.3g}")
    print(f"  Min: {completed_df[parameter_name].min():.3g}")
    print(f"  Max: {completed_df[parameter_name].max():.3g}")
    
    # Parameter analysis
    print(f"\n{parameter_name.replace('_', ' ').title()} Analysis:")
    param_sorted = completed_df.sort_values('best_val_loss')
    print(f"  Top 3 {parameter_name.replace('_', ' ').title()} Values (by validation loss):")
    for i, (_, row) in enumerate(param_sorted.head(3).iterrows()):
        print(f"    {i+1}. {parameter_name}: {row[parameter_name]:.3g}, Val Loss: {row['best_val_loss']:.4f}")
    
    # Convergence analysis (if epochs data available)
    if 'epochs' in completed_df.columns and 'best_epoch' in completed_df.columns:
        completed_df['epochs'] = pd.to_numeric(completed_df['epochs'], errors='coerce')
        completed_df['best_epoch'] = pd.to_numeric(completed_df['best_epoch'], errors='coerce')
        
        print(f"\nConvergence Analysis:")
        print(f"  Average epochs: {completed_df['epochs'].mean():.1f}")
        print(f"  Average best epoch: {completed_df['best_epoch'].mean():.1f}")
        
        # Early stopping analysis
        early_stop_ratio = completed_df['best_epoch'] / completed_df['epochs']
        print(f"  Early stopping ratio (best_epoch/total_epochs): {early_stop_ratio.mean():.2f}")
    
    # Duration analysis (if available)
    if 'duration_minutes' in completed_df.columns:
        completed_df['duration_minutes'] = pd.to_numeric(completed_df['duration_minutes'], errors='coerce')
        print(f"\nDuration Analysis:")
        print(f"  Average duration: {completed_df['duration_minutes'].mean():.1f} minutes")
        print(f"  Total duration: {completed_df['duration_minutes'].sum():.1f} minutes")
        print(f"  Fastest experiment: {completed_df['duration_minutes'].min():.1f} minutes")
        print(f"  Slowest experiment: {completed_df['duration_minutes'].max():.1f} minutes")
    
    # Failed experiments
    if not failed_df.empty:
        print(f"\nFailed Experiments:")
        failed_df[parameter_name] = pd.to_numeric(failed_df[parameter_name], errors='coerce')
        for _, row in failed_df.iterrows():
            print(f"  {parameter_name}: {row[parameter_name]:.3g}")
    
    # Recommendations
    print(f"\n{parameter_name.replace('_', ' ').title()} Recommendations:")
    best_param = completed_df.loc[best_val_idx, parameter_name]
    
    # Check if best value is at the boundary
    param_values = sorted(completed_df[parameter_name].unique())
    if best_param == param_values[0]:
        print(f"  ⚠️  Best value is at the lower boundary. Consider testing lower values.")
    elif best_param == param_values[-1]:
        print(f"  ⚠️  Best value is at the upper boundary. Consider testing higher values.")
    else:
        print(f"  ✅ Best value is not at the boundary, search range seems appropriate.")
    
    # Check for clear trend
    correlation = completed_df[parameter_name].corr(completed_df['best_val_loss'])
    if abs(correlation) > 0.7:
        trend = "positive" if correlation > 0 else "negative"
        print(f"  📈 Strong {trend} correlation ({correlation:.3f}) between {parameter_name} and validation loss.")
    else:
        print(f"  📊 No strong linear correlation ({correlation:.3f}) between {parameter_name} and validation loss.")

def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter search results')
    parser.add_argument('csv_file', help='Path to the results CSV file')
    parser.add_argument('--output-dir', default='.', help='Output directory for plots')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Load data and get parameter name
    df, parameter_name = load_and_validate_data(args.csv_file)
    
    # Print detailed analysis
    print_detailed_analysis(df, parameter_name)
    
    # Create plots
    if not args.no_plots:
        try:
            create_analysis_plots(df, parameter_name, args.output_dir)
        except ImportError as e:
            print(f"Warning: Could not create plots due to missing dependencies: {e}")
            print("Install matplotlib and seaborn to enable plotting")
        except Exception as e:
            print(f"Error creating plots: {e}")

if __name__ == "__main__":
    main()
