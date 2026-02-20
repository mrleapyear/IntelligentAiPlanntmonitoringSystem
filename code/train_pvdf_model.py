"""
Train AI model for PVDF-based plant health monitoring
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class PVDFModelTrainer:
    def __init__(self, data_path=None):
        """
        Initialize model trainer
        
        Parameters:
        -----------
        data_path : str, optional
            Path to training data CSV file
        """
        self.data_path = data_path
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
        # Create directories
        os.makedirs('../models', exist_ok=True)
        os.makedirs('../logs', exist_ok=True)
        
        # Class names
        self.class_names = ['Healthy', 'Pest Stress', 'Water Stress']
        
        # Colors for visualization
        self.colors = ['#4CAF50', '#FF9800', '#2196F3']
    
    def load_data(self, data_path=None):
        """
        Load training data from CSV file
        
        Parameters:
        -----------
        data_path : str, optional
            Path to CSV file. If None, uses self.data_path
        """
        if data_path:
            self.data_path = data_path
        
        if not self.data_path:
            # Find the latest data file
            data_dir = '../data'
            csv_files = [f for f in os.listdir(data_dir) if f.startswith('pvdf_plant_data') and f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No training data found. Run create_pvdf_data.py first.")
            
            # Use the most recent file
            self.data_path = os.path.join(data_dir, sorted(csv_files)[-1])
            print(f"📁 Using data file: {self.data_path}")
        
        # Load data
        print(f"📊 Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col != 'health_status']
        self.X = df[feature_columns].values
        self.y = df['health_status'].values
        
        print(f"✅ Data loaded successfully!")
        print(f"   Samples: {len(self.X)}")
        print(f"   Features: {self.X.shape[1]}")
        print(f"   Classes: {len(np.unique(self.y))}")
        print(f"   Feature names: {feature_columns}")
        
        # Show class distribution
        unique, counts = np.unique(self.y, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"   {self.class_names[cls]}: {count} samples")
        
        return df
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess and split data
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        """
        print("\n🔧 Preprocessing data...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data preprocessing complete!")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Testing samples: {len(self.X_test)}")
        print(f"   Feature scaling applied")
    
    def train_models(self):
        """
        Train multiple models and select the best one
        """
        print("\n🤖 Training AI models...")
        print("=" * 60)
        
        # Define models to train
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
        }
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_train = model.predict(self.X_train_scaled)
            
            # Calculate accuracy
            test_accuracy = accuracy_score(self.y_test, y_pred)
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store model
            self.models[name] = {
                'model': model,
                'test_accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'y_pred': y_pred
            }
            
            print(f"  Training Accuracy: {train_accuracy:.2%}")
            print(f"  Testing Accuracy:  {test_accuracy:.2%}")
            print(f"  CV Accuracy:       {cv_mean:.2%} (+/- {cv_std:.2%})")
            
            # Update best model
            if test_accuracy > self.best_score:
                self.best_score = test_accuracy
                self.best_model = model
                self.best_model_name = name
        
        print("\n" + "=" * 60)
        print(f"🏆 Best Model: {self.best_model_name}")
        print(f"   Test Accuracy: {self.best_score:.2%}")
        print("=" * 60)
    
    def evaluate_models(self):
        """
        Evaluate and compare all trained models
        """
        print("\n📊 MODEL EVALUATION")
        print("=" * 60)
        
        # Create comparison table
        comparison = []
        for name, results in self.models.items():
            comparison.append({
                'Model': name,
                'Train Acc': f"{results['train_accuracy']:.2%}",
                'Test Acc': f"{results['test_accuracy']:.2%}",
                'CV Mean': f"{results['cv_mean']:.2%}",
                'CV Std': f"{results['cv_std']:.2%}"
            })
        
        df_comparison = pd.DataFrame(comparison)
        print("\nModel Comparison:")
        print(df_comparison.to_string(index=False))
        
        # Detailed evaluation of best model
        print(f"\n📋 Detailed Report for {self.best_model_name}:")
        y_pred = self.models[self.best_model_name]['y_pred']
        
        # Classification report
        report = classification_report(self.y_test, y_pred, 
                                      target_names=self.class_names,
                                      output_dict=True)
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Feature importance (for Random Forest)
        if self.best_model_name == 'Random Forest':
            print("\n🔍 Feature Importance:")
            feature_names = [col for col in pd.read_csv(self.data_path).columns if col != 'health_status']
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i in range(len(feature_names)):
                print(f"  {i+1:2}. {feature_names[indices[i]]:20} {importances[indices[i]]:.4f}")
        
        return report, cm
    
    def visualize_results(self, cm):
        """
        Create visualizations of model performance
        """
        print("\n📈 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Model Comparison
        ax1 = axes[0, 0]
        models = list(self.models.keys())
        test_accs = [self.models[m]['test_accuracy'] for m in models]
        
        bars = ax1.bar(models, test_accs, color=self.colors[:len(models)])
        ax1.set_title('Model Test Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim([0, 1])
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, acc in zip(bars, test_accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom')
        
        # Plot 2: Confusion Matrix
        ax2 = axes[0, 1]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax2)
        ax2.set_title('Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # Plot 3: Feature Importance (if Random Forest)
        ax3 = axes[0, 2]
        if self.best_model_name == 'Random Forest':
            feature_names = [col for col in pd.read_csv(self.data_path).columns if col != 'health_status']
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 features
            
            ax3.barh(range(len(indices)), importances[indices], color='#4CAF50')
            ax3.set_yticks(range(len(indices)))
            ax3.set_yticklabels([feature_names[i] for i in indices])
            ax3.set_title('Top 10 Feature Importances')
            ax3.set_xlabel('Importance')
        else:
            ax3.text(0.5, 0.5, 'Feature importance\nonly available for\nRandom Forest',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Feature Importance')
        
        # Plot 4: Learning Curves (simulated)
        ax4 = axes[1, 0]
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Simulate learning curve (in practice, use learning_curve from sklearn)
        train_scores = []
        test_scores = []
        
        for size in train_sizes:
            n_samples = int(size * len(self.X_train))
            idx = np.random.choice(len(self.X_train), n_samples, replace=False)
            
            # Train on subset
            model_temp = RandomForestClassifier(n_estimators=50, random_state=42)
            model_temp.fit(self.X_train_scaled[idx], self.y_train[idx])
            
            # Score
            train_scores.append(model_temp.score(self.X_train_scaled[idx], self.y_train[idx]))
            test_scores.append(model_temp.score(self.X_test_scaled, self.y_test))
        
        ax4.plot(train_sizes * len(self.X_train), train_scores, 'o-', label='Training', linewidth=2)
        ax4.plot(train_sizes * len(self.X_train), test_scores, 's-', label='Testing', linewidth=2)
        ax4.set_title('Learning Curve')
        ax4.set_xlabel('Training Samples')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Class Distribution
        ax5 = axes[1, 1]
        unique_train, counts_train = np.unique(self.y_train, return_counts=True)
        unique_test, counts_test = np.unique(self.y_test, return_counts=True)
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        ax5.bar(x - width/2, counts_train, width, label='Train', color=self.colors[0])
        ax5.bar(x + width/2, counts_test, width, label='Test', color=self.colors[1])
        
        ax5.set_title('Class Distribution')
        ax5.set_xlabel('Class')
        ax5.set_ylabel('Number of Samples')
        ax5.set_xticks(x)
        ax5.set_xticklabels(self.class_names)
        ax5.legend()
        
        # Add count labels
        for i, (train_count, test_count) in enumerate(zip(counts_train, counts_test)):
            ax5.text(i - width/2, train_count + 5, str(train_count), ha='center')
            ax5.text(i + width/2, test_count + 5, str(test_count), ha='center')
        
        # Plot 6: ROC Curves (simulated)
        ax6 = axes[1, 2]
        # Simulate ROC curves for each class
        for i, class_name in enumerate(self.class_names):
            # Generate random probabilities for demonstration
            if hasattr(self.best_model, 'predict_proba'):
                y_proba = self.best_model.predict_proba(self.X_test_scaled)[:, i]
                # Simple simulated ROC points
                thresholds = np.linspace(0, 1, 50)
                tpr = []
                fpr = []
                
                for thresh in thresholds:
                    y_pred_thresh = (y_proba >= thresh).astype(int)
                    tp = np.sum((y_pred_thresh == 1) & (self.y_test == i))
                    fp = np.sum((y_pred_thresh == 1) & (self.y_test != i))
                    tn = np.sum((y_pred_thresh == 0) & (self.y_test != i))
                    fn = np.sum((y_pred_thresh == 0) & (self.y_test == i))
                    
                    if tp + fn > 0:
                        tpr.append(tp / (tp + fn))
                    else:
                        tpr.append(0)
                    
                    if fp + tn > 0:
                        fpr.append(fp / (fp + tn))
                    else:
                        fpr.append(0)
                
                ax6.plot(fpr, tpr, label=f'{class_name}', linewidth=2)
        
        ax6.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line
        ax6.set_title('ROC Curves (Simulated)')
        ax6.set_xlabel('False Positive Rate')
        ax6.set_ylabel('True Positive Rate')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_path = f'../logs/model_performance_{timestamp}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved to: {fig_path}")
    
    def save_model(self):
        """
        Save the trained model and scaler
        """
        if self.best_model is None:
            print("❌ No model trained yet!")
            return
        
        # Create timestamp for versioning
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = f'../models/pvdf_plant_model_{timestamp}.pkl'
        joblib.dump(self.best_model, model_path)
        
        # Save scaler
        scaler_path = f'../models/scaler_{timestamp}.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # Create symlinks for latest model
        latest_model_path = '../models/pvdf_plant_model_latest.pkl'
        latest_scaler_path = '../models/scaler_latest.pkl'
        
        # Remove existing symlinks
        for path in [latest_model_path, latest_scaler_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # Create new symlinks
        os.symlink(model_path, latest_model_path)
        os.symlink(scaler_path, latest_scaler_path)
        
        # Save training metadata
        metadata = {
            'timestamp': timestamp,
            'model_name': self.best_model_name,
            'test_accuracy': float(self.best_score),
            'training_date': datetime.now().isoformat(),
            'data_file': self.data_path,
            'n_samples': len(self.X),
            'n_features': self.X.shape[1],
            'feature_names': [col for col in pd.read_csv(self.data_path).columns if col != 'health_status'],
            'class_names': self.class_names,
            'model_type': type(self.best_model).__name__,
            'model_params': self.best_model.get_params() if hasattr(self.best_model, 'get_params') else {}
        }
        
        metadata_path = f'../models/model_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create symlink for metadata
        latest_metadata_path = '../models/model_metadata_latest.json'
        if os.path.exists(latest_metadata_path):
            os.remove(latest_metadata_path)
        os.symlink(metadata_path, latest_metadata_path)
        
        print("\n💾 MODEL SAVED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Scaler: {scaler_path}")
        print(f"Metadata: {metadata_path}")
        print(f"Best accuracy: {self.best_score:.2%}")
        print("=" * 60)
        
        return model_path, scaler_path, metadata_path

def main():
    """Main training function"""
    print("🌿 PVDF PLANT HEALTH AI - MODEL TRAINING")
    print("=" * 60)
    
    # Get user input
    data_path = input("Enter path to training data [press Enter for latest]: ").strip()
    if not data_path:
        data_path = None
    
    try:
        # Initialize trainer
        trainer = PVDFModelTrainer(data_path)
        
        # Load data
        df = trainer.load_data()
        
        # Preprocess data
        trainer.preprocess_data()
        
        # Train models
        trainer.train_models()
        
        # Evaluate models
        report, cm = trainer.evaluate_models()
        
        # Visualize results
        visualize = input("\nCreate visualizations? (y/n): ").lower()
        if visualize == 'y':
            trainer.visualize_results(cm)
        
        # Save model
        save = input("\nSave trained model? (y/n): ").lower()
        if save == 'y':
            model_path, scaler_path, metadata_path = trainer.save_model()
            
            print("\n🎯 NEXT STEPS:")
            print("1. Use the saved model in pvdf_monitor.py")
            print("2. Test with real sensor data")
            print("3. Deploy with web_dashboard.py")
            print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()