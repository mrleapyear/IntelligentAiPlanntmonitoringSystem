"""
Generate training data for PVDF-based plant health AI model
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json

class PVDFDataGenerator:
    def __init__(self, samples_per_class=150, features=8):
        """
        Initialize data generator
        
        Parameters:
        -----------
        samples_per_class : int
            Number of samples per health status
        features : int
            Number of features to generate
        """
        self.samples_per_class = samples_per_class
        self.features = features
        self.data = []
        self.labels = []
        
        # Feature ranges based on PVDF sensor characteristics
        self.feature_ranges = {
            'healthy': {
                'rms': (10, 50),        # mV
                'peak_freq': (15, 35),   # Hz
                'zero_crossing': (20, 40),  # %
                'peak_peak': (30, 100),  # mV
                'mean_abs': (8, 25),     # mV
                'skewness': (-0.5, 0.5),
                'kurtosis': (-1, 1),
                'energy': (50, 200)      # arbitrary units
            },
            'pest_stress': {
                'rms': (50, 150),        # Higher vibration
                'peak_freq': (40, 80),   # Higher frequency
                'zero_crossing': (40, 70),  # More zero crossings
                'peak_peak': (100, 300), # Larger amplitude
                'mean_abs': (25, 75),    # Higher mean
                'skewness': (0.5, 2),    # Positive skew
                'kurtosis': (1, 3),      # More peaked
                'energy': (200, 500)     # Higher energy
            },
            'water_stress': {
                'rms': (1, 10),          # Very low vibration
                'peak_freq': (5, 20),    # Lower frequency
                'zero_crossing': (10, 30),  # Fewer zero crossings
                'peak_peak': (10, 50),   # Smaller amplitude
                'mean_abs': (1, 8),      # Lower mean
                'skewness': (-2, -0.5),  # Negative skew
                'kurtosis': (-2, 0),     # Flatter distribution
                'energy': (10, 100)      # Lower energy
            }
        }
        
        # Create data directory if it doesn't exist
        os.makedirs('../data', exist_ok=True)
        os.makedirs('../logs', exist_ok=True)
    
    def generate_class_data(self, class_name, label):
        """
        Generate data for a specific class
        
        Parameters:
        -----------
        class_name : str
            Name of the class ('healthy', 'pest_stress', 'water_stress')
        label : int
            Numerical label for the class (0, 1, 2)
        """
        print(f"Generating {self.samples_per_class} samples for {class_name}...")
        
        ranges = self.feature_ranges[class_name]
        
        for _ in range(self.samples_per_class):
            # Generate features with some randomness
            sample = [
                np.random.uniform(*ranges['rms']),
                np.random.uniform(*ranges['peak_freq']),
                np.random.uniform(*ranges['zero_crossing']),
                np.random.uniform(*ranges['peak_peak']),
                np.random.uniform(*ranges['mean_abs']),
                np.random.uniform(*ranges['skewness']),
                np.random.uniform(*ranges['kurtosis']),
                np.random.uniform(*ranges['energy'])
            ]
            
            # Add some noise and correlation
            sample = self.add_correlations(sample, class_name)
            
            self.data.append(sample)
            self.labels.append(label)
    
    def add_correlations(self, sample, class_name):
        """
        Add realistic correlations between features
        """
        sample = np.array(sample)
        
        # Healthy plants: balanced correlations
        if class_name == 'healthy':
            # RMS and energy should correlate
            sample[7] = sample[0] * 4 + np.random.normal(0, 10)
            # Frequency and zero crossing correlate
            sample[2] = sample[1] * 0.8 + np.random.normal(0, 5)
        
        # Pest stress: high vibration correlates with high frequency
        elif class_name == 'pest_stress':
            sample[1] = sample[0] * 0.5 + np.random.normal(20, 5)
            sample[2] = sample[0] * 0.3 + np.random.normal(20, 3)
        
        # Water stress: low everything
        elif class_name == 'water_stress':
            sample[1] = sample[0] * 1.5 + np.random.normal(5, 2)
            sample[3] = sample[0] * 3 + np.random.normal(5, 5)
        
        # Ensure all values are positive (except skewness/kurtosis)
        for i in [0, 2, 3, 4, 7]:  # Positive features
            sample[i] = abs(sample[i])
        
        return sample.tolist()
    
    def save_metadata(self, filepath):
        """Save dataset metadata"""
        metadata = {
            'created': datetime.now().isoformat(),
            'total_samples': len(self.data),
            'samples_per_class': self.samples_per_class,
            'features': self.features,
            'feature_names': [
                'rms_mv', 'peak_freq_hz', 'zero_crossing_percent',
                'peak_peak_mv', 'mean_abs_mv', 'skewness',
                'kurtosis', 'energy'
            ],
            'class_names': ['healthy', 'pest_stress', 'water_stress'],
            'class_distribution': {
                'healthy': self.samples_per_class,
                'pest_stress': self.samples_per_class,
                'water_stress': self.samples_per_class
            },
            'feature_ranges': self.feature_ranges
        }
        
        with open(filepath.replace('.csv', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print("=" * 60)
        print("🌿 GENERATING PLANT HEALTH TRAINING DATA")
        print("=" * 60)
        
        # Generate data for each class
        self.generate_class_data('healthy', 0)
        self.generate_class_data('pest_stress', 1)
        self.generate_class_data('water_stress', 2)
        
        # Convert to numpy arrays
        X = np.array(self.data)
        y = np.array(self.labels)
        
        # Create DataFrame
        feature_names = [
            'rms_mv', 'peak_freq_hz', 'zero_crossing_percent',
            'peak_peak_mv', 'mean_abs_mv', 'skewness',
            'kurtosis', 'energy', 'health_status'
        ]
        
        df = pd.DataFrame(X, columns=feature_names[:-1])
        df['health_status'] = y
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'../data/pvdf_plant_data_{timestamp}.csv'
        df.to_csv(filename, index=False)
        
        # Save metadata
        self.save_metadata(filename)
        
        print("\n" + "=" * 60)
        print("✅ DATASET GENERATION COMPLETE!")
        print("=" * 60)
        print(f"File saved: {filename}")
        print(f"Total samples: {len(df)}")
        print(f"Feature shape: {X.shape}")
        print("\nClass distribution:")
        print(f"  Healthy plants: {(y == 0).sum()}")
        print(f"  Pest stress: {(y == 1).sum()}")
        print(f"  Water stress: {(y == 2).sum()}")
        
        # Show sample data
        print("\n📊 Sample data (first 5 rows):")
        print(df.head())
        
        return df
    
    def analyze_dataset(self, df):
        """Analyze and visualize the generated dataset"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("\n📈 DATASET ANALYSIS")
        print("=" * 40)
        
        # Basic statistics
        print("\n📊 Feature Statistics:")
        print(df.describe())
        
        # Class distribution visualization
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Class distribution
        plt.subplot(2, 2, 1)
        class_counts = df['health_status'].value_counts()
        colors = ['#4CAF50', '#FF9800', '#2196F3']
        class_names = ['Healthy', 'Pest Stress', 'Water Stress']
        bars = plt.bar(class_names, class_counts.values, color=colors)
        plt.title('Class Distribution')
        plt.ylabel('Number of Samples')
        
        # Add count labels on bars
        for bar, count in zip(bars, class_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom')
        
        # Plot 2: Feature distributions
        plt.subplot(2, 2, 2)
        features_to_plot = ['rms_mv', 'peak_freq_hz', 'peak_peak_mv']
        for i, feature in enumerate(features_to_plot):
            plt.hist(df[feature], alpha=0.5, label=feature, bins=30)
        plt.title('Feature Distributions')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot 3: Correlation heatmap
        plt.subplot(2, 2, 3)
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        
        # Plot 4: Pair plot sample
        plt.subplot(2, 2, 4)
        sample_df = df.sample(min(100, len(df)))
        for status, color in zip([0, 1, 2], colors):
            status_data = sample_df[sample_df['health_status'] == status]
            plt.scatter(status_data['rms_mv'], status_data['peak_freq_hz'],
                       color=color, alpha=0.6, label=class_names[status])
        plt.title('RMS vs Peak Frequency')
        plt.xlabel('RMS (mV)')
        plt.ylabel('Peak Frequency (Hz)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('../data/dataset_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Analysis saved to: ../data/dataset_analysis.png")

def main():
    """Main function"""
    print("🌿 PVDF Plant Health AI - Training Data Generator")
    print("=" * 60)
    
    # Get user input
    try:
        samples = int(input("Enter samples per class [default: 150]: ") or "150")
    except ValueError:
        samples = 150
        print("Using default: 150 samples per class")
    
    # Generate dataset
    generator = PVDFDataGenerator(samples_per_class=samples)
    df = generator.generate_dataset()
    
    # Ask if user wants analysis
    analyze = input("\nPerform dataset analysis? (y/n): ").lower()
    if analyze == 'y':
        generator.analyze_dataset(df)
    
    print("\n🎯 Next steps:")
    print("1. Review the generated data in ../data/")
    print("2. Run train_pvdf_model.py to train AI model")
    print("3. Use calibrate_pvdf.py for sensor calibration")
    print("=" * 60)

if __name__ == "__main__":
    main()