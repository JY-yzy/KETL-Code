#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Module - Provides various plotting functions
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib settings
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False


def ensure_output_dir():
    """Ensure output directory exists"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_figure(fig, filename):
    """
    Save figure as PNG and PDF formats
    
    Args:
        fig: matplotlib figure object
        filename: filename without extension
    """
    output_dir = ensure_output_dir()
    
    # Save as PNG
    png_path = os.path.join(output_dir, f'{filename}.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Save as PDF (publication-quality)
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    
    print(f"Figure saved to: {png_path}")
    print(f"Figure saved to: {pdf_path}")


def plot_training_curves(log_file):
    """
    Plot training curves: loss and F1 curves
    
    Args:
        log_file: path to log file (CSV format)
    """
    # Read log file
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        return
    
    df = pd.read_csv(log_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot loss curves
    axes[0].plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o', color='#1f77b4')
    if 'val_loss' in df.columns:
        axes[0].plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s', color='#ff7f0e')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curves')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot F1 curves
    if 'ner_micro_f1' in df.columns:
        axes[1].plot(df['epoch'], df['ner_micro_f1'] * 100, label='NER Micro-F1', marker='o', color='#2ca02c')
    if 'ner_macro_f1' in df.columns:
        axes[1].plot(df['epoch'], df['ner_macro_f1'] * 100, label='NER Macro-F1', marker='s', color='#9467bd')
    if 'rel_micro_f1' in df.columns:
        axes[1].plot(df['epoch'], df['rel_micro_f1'] * 100, label='RE Micro-F1', marker='^', color='#d62728')
    if 'event_micro_f1' in df.columns:
        axes[1].plot(df['epoch'], df['event_micro_f1'] * 100, label='EE Micro-F1', marker='v', color='#8c564b')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 (%)')
    axes[1].set_title('Model Performance Curves')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save figures (PNG and PDF)
    save_figure(fig, 'training_curves')
    plt.close()


def plot_ablation_study(csv_path):
    """
    Plot ablation study bar chart
    
    Args:
        csv_path: path to ablation study results CSV file
    """
    # Read data
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Get dataset names
    dataset_columns = [col for col in df.columns if '_micro_f1' in col]
    datasets = [col.replace('_micro_f1', '') for col in dataset_columns]
    
    # Set colors (extended color list for more datasets)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    # Plot bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.15
    x = np.arange(len(df))
    
    for i, dataset in enumerate(datasets):
        scores = df[f'{dataset}_micro_f1'] * 100
        ax.bar(x + i * bar_width, scores, width=bar_width, label=dataset, color=colors[i])
    
    ax.set_xlabel('Ablation Configuration')
    ax.set_ylabel('Micro-F1 (%)')
    ax.set_title('Ablation Study Results')
    ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(df['config'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 100)
    
    # Add value labels
    for i, dataset in enumerate(datasets):
        scores = df[f'{dataset}_micro_f1'] * 100
        for j, score in enumerate(scores):
            ax.text(x[j] + i * bar_width, score + 1, f'{score:.1f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save figures (PNG and PDF)
    save_figure(fig, 'ablation_study')
    plt.close()


def plot_frequency_analysis(csv_path):
    """
    Plot frequency group bar chart
    
    Args:
        csv_path: path to frequency analysis CSV file
    """
    # Read data
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Assume data format: category, group, frequency
    fig, ax = plt.subplots(figsize=(12, 6))
    
    groups = df['group'].unique()
    categories = df['category'].unique()
    
    bar_width = 0.2
    x = np.arange(len(categories))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, group in enumerate(groups[:4]):
        group_data = df[df['group'] == group]
        ax.bar(x + i * bar_width, group_data['frequency'], width=bar_width, label=group, color=colors[i])
    
    ax.set_xlabel('Category')
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency Group Analysis')
    ax.set_xticks(x + bar_width * (len(groups[:4]) - 1) / 2)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figures (PNG and PDF)
    save_figure(fig, 'frequency_analysis')
    plt.close()


def plot_frequency_bar(csv_path):
    """
    Plot frequency-level F1 comparison chart (high/medium/low frequency)
    
    Args:
        csv_path: path to frequency-level evaluation CSV file with format:
            frequency_level (high/medium/low), f1_score
    """
    # Read data
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Prepare data
    frequency_levels = ['high', 'medium', 'low']
    frequency_labels = ['High', 'Medium', 'Low']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Calculate average F1 for each frequency level
    f1_scores = []
    for level in frequency_levels:
        level_data = df[df['frequency_level'] == level]
        if len(level_data) > 0:
            # Try multiple possible F1 column names
            if 'f1_score' in level_data.columns:
                avg_f1 = level_data['f1_score'].mean()
            elif 'micro_f1' in level_data.columns:
                avg_f1 = level_data['micro_f1'].mean()
            elif 'macro_f1' in level_data.columns:
                avg_f1 = level_data['macro_f1'].mean()
            else:
                avg_f1 = 0.0
        else:
            avg_f1 = 0.0
        f1_scores.append(avg_f1 * 100)  # Convert to percentage
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(frequency_labels, f1_scores, color=colors, width=0.6)
    
    # Set figure properties
    ax.set_ylabel('F1 Score (%)')
    ax.set_title('NER Performance by Frequency Level')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{score:.1f}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    # Save figures (PNG and PDF)
    save_figure(fig, 'frequency_bar')
    plt.close()


def plot_geoglue_radar(csv_path):
    """
    Plot GeoGLUE radar chart
    
    Args:
        csv_path: path to GeoGLUE test results CSV file
    """
    # Read data
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Get task names and scores
    # Try multiple possible column names
    if 'task' in df.columns and 'f1' in df.columns:
        tasks = df['task'].tolist()
        scores = df['f1'].tolist()
    elif 'task' in df.columns and 'score' in df.columns:
        tasks = df['task'].tolist()
        scores = df['score'].tolist()
    else:
        print(f"Error: Unrecognized CSV format, columns: {df.columns.tolist()}")
        return
    
    # Radar chart configuration
    num_vars = len(tasks)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    scores_normalized = [s * 100 for s in scores]
    scores_normalized += scores_normalized[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    
    # Plot radar chart
    ax.fill(angles, scores_normalized, color='#1f77b4', alpha=0.3)
    ax.plot(angles, scores_normalized, color='#1f77b4', linewidth=2, marker='o')
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, fontsize=10)
    
    # Set radial ticks
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'])
    
    ax.set_title('GeoGLUE Zero-shot Results', pad=20)
    
    # Add value labels
    for angle, score, task in zip(angles[:-1], scores, tasks):
        ax.text(angle, score * 100 + 5, f'{score*100:.1f}', ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save figures (PNG and PDF)
    save_figure(fig, 'geoglue_radar')
    plt.close()


def plot_ner_results(csv_path):
    """
    Plot NER comparison experiment results
    
    Args:
        csv_path: path to NER results CSV file
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get test datasets
    test_datasets = ['weibo_ner', 'cmner', 'cluener', 'msra']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bar_width = 0.2
    x = np.arange(len(df))
    
    for i, dataset in enumerate(test_datasets):
        scores = df[f'{dataset}_micro_f1'] * 100
        ax.bar(x + i * bar_width, scores, width=bar_width, label=dataset, color=colors[i])
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Micro-F1 (%)')
    ax.set_title('NER Model Comparison Results')
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(df['model_name'], rotation=60, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(70, 100)
    
    plt.tight_layout()
    
    # Save figures (PNG and PDF)
    save_figure(fig, 'ner_results')
    plt.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualization Tool')
    
    parser.add_argument('--plot_type', type=str, required=True,
                        choices=['training_curves', 'ablation_study', 'frequency_analysis', 
                                 'frequency_bar', 'geoglue_radar', 'ner_results'],
                        help='Plot type')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input file path')
    
    return parser.parse_args()


def main(args):
    """Main function"""
    plot_type = args.get('plot_type', args.get('--plot_type', None))
    plot_input = args.get('plot_input', args.get('--input', None))
    
    if not plot_type or not plot_input:
        print("Error: Missing required arguments")
        return
    
    if plot_type == 'training_curves':
        plot_training_curves(plot_input)
    elif plot_type == 'ablation_study':
        plot_ablation_study(plot_input)
    elif plot_type == 'frequency_analysis':
        plot_frequency_analysis(plot_input)
    elif plot_type == 'frequency_bar':
        plot_frequency_bar(plot_input)
    elif plot_type == 'geoglue_radar':
        plot_geoglue_radar(plot_input)
    elif plot_type == 'ner_results':
        plot_ner_results(plot_input)
    else:
        print(f"Error: Unknown plot type: {plot_type}")


if __name__ == '__main__':
    args = parse_args()
    main({'plot_type': args.plot_type, 'plot_input': args.input})
