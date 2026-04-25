"""
BERT-based Nutrition Information Classifier
============================================

This module provides transformer-based classification for detecting whether
nutritional information is present in veterinary clinical text.

Supports two pre-trained models:
1. Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) - Clinical NLP
2. VetBERT (havocy28/VetBERT) - Veterinary-specific

Key features:
- Fine-tuning on annotated veterinary nutrition data
- Cross-validation for robust evaluation
- Hyperparameter tuning support
- Early stopping to prevent overfitting
- Model comparison utilities

Author: Improved Pipeline
Date: 2024
"""

import os
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Transformers imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Available pre-trained models
AVAILABLE_MODELS = {
    'bio_clinical_bert': 'emilyalsentzer/Bio_ClinicalBERT',
    'vet_bert': 'havocy28/VetBERT',
    'pubmed_bert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    'bert_base': 'bert-base-uncased',
}


@dataclass
class ClassifierConfig:
    """Configuration for the BERT classifier."""
    model_name: str = 'bio_clinical_bert'
    max_length: int = 512
    batch_size: int = 16 // 2
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    early_stopping_patience: int = 3
    output_dir: str = './bert_output'


class NutritionDataset(Dataset):
    """PyTorch Dataset for nutrition classification."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Any = None,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] is not None else ""

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }

        # Add token_type_ids if present (some models need this)
        # if 'token_type_ids' in encoding:
        #     item['token_type_ids'] = encoding['token_type_ids'].squeeze()

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }


class BERTNutritionClassifier:
    """
    BERT-based classifier for detecting nutritional information in veterinary text.
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        """
        Initialize the classifier.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or ClassifierConfig()
        self.model = None
        self.tokenizer = None
        self.label_map = {'No': 0, 'Yes': 1}
        self.reverse_label_map = {0: 'No', 1: 'Yes'}

        set_seed(self.config.seed)

    def _get_model_path(self) -> str:
        """Get the HuggingFace model path."""
        if self.config.model_name in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[self.config.model_name]
        return self.config.model_name

    def load_model(self):
        """Load the pre-trained model and tokenizer."""
        model_path = self._get_model_path()
        logger.info(f"Loading model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            ignore_mismatched_sizes=True
        )

        logger.info(f"Model loaded successfully")

    def prepare_data(
        self,
        df: pd.DataFrame,
        text_column: str = 'ExaminationText',
        label_column: str = 'Annotation',
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training.

        Args:
            df: DataFrame with text and labels
            text_column: Column name for text
            label_column: Column name for labels
            test_size: Fraction for test set

        Returns:
            Tuple of (train_df, test_df)
        """
        # Filter to annotated records
        df = df[df[label_column].notna()].copy()

        # Normalize labels
        df['label'] = df[label_column].str.strip().str.capitalize()
        df['label_id'] = df['label'].map(self.label_map)

        # Stratified split
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['label_id'],
            random_state=self.config.seed
        )

        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        logger.info(f"Train label distribution:\n{train_df['label'].value_counts()}")
        logger.info(f"Test label distribution:\n{test_df['label'].value_counts()}")

        return train_df, test_df

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        text_column: str = 'ExaminationText'
    ):
        """
        Fine-tune the model on training data.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            text_column: Column name for text
        """
        if self.model is None:
            self.load_model()

        # Create datasets
        train_dataset = NutritionDataset(
            texts=train_df[text_column].tolist(),
            labels=train_df['label_id'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )

        val_dataset = NutritionDataset(
            texts=val_df[text_column].tolist(),
            labels=val_df['label_id'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_dir=f'{self.config.output_dir}/logs',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,  # Only keep the best checkpoint, delete others
            # gradient_accumulation_steps=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=self.config.seed,
            report_to="none",  # Disable wandb etc.
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            ]
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed!")

        # Evaluate on validation set
        eval_results = trainer.evaluate()
        logger.info(f"Validation results: {eval_results}")

        return trainer, eval_results

    def predict(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Tuple[List[str], List[float]]:
        """
        Make predictions on new texts.

        Args:
            texts: List of texts to classify
            batch_size: Batch size for inference

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")

        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        all_predictions = []
        all_probabilities = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )

            encodings = {k: v.to(device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)

            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            probabilities = probs[:, 1].cpu().numpy()  # Probability of "Yes"

            all_predictions.extend([self.reverse_label_map[p] for p in predictions])
            all_probabilities.extend(probabilities.tolist())

        return all_predictions, all_probabilities

    def evaluate(
        self,
        test_df: pd.DataFrame,
        text_column: str = 'ExaminationText',
        label_column: str = 'label'
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            test_df: Test DataFrame
            text_column: Column name for text
            label_column: Column name for labels

        Returns:
            Dictionary of metrics
        """
        predictions, probabilities = self.predict(test_df[text_column].tolist())

        y_true = test_df[label_column].tolist()
        y_pred = predictions

        # Convert to numeric for sklearn
        y_true_numeric = [self.label_map[y] for y in y_true]
        y_pred_numeric = [self.label_map[y] for y in y_pred]

        accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
        precision = precision_score(y_true_numeric, y_pred_numeric)
        recall = recall_score(y_true_numeric, y_pred_numeric)
        f1 = f1_score(y_true_numeric, y_pred_numeric)

        tn, fp, fn, tp = confusion_matrix(y_true_numeric, y_pred_numeric).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }

    def save_model(self, path: str):
        """Save the model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def load_trained_model(self, path: str):
        """Load a previously trained model."""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        logger.info(f"Model loaded from {path}")


def cross_validate(
    df: pd.DataFrame,
    config: ClassifierConfig,
    text_column: str = 'ExaminationText',
    label_column: str = 'Annotation',
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation.

    Args:
        df: DataFrame with data
        config: Classifier configuration
        text_column: Column name for text
        label_column: Column name for labels
        n_folds: Number of folds

    Returns:
        Dictionary with fold results and aggregate metrics
    """
    # Prepare data
    df = df[df[label_column].notna()].copy()
    df['label'] = df[label_column].str.strip().str.capitalize()
    df['label_id'] = df['label'].map({'No': 0, 'Yes': 1})

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label_id'])):
        logger.info(f"\n{'='*50}")
        logger.info(f"FOLD {fold + 1}/{n_folds}")
        logger.info(f"{'='*50}")

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # Create fresh classifier for each fold
        classifier = BERTNutritionClassifier(config)
        classifier.load_model()

        # Train
        fold_config = ClassifierConfig(**vars(config))
        fold_config.output_dir = f"{config.output_dir}/fold_{fold+1}"
        classifier.config = fold_config

        trainer, _ = classifier.train(train_df, val_df, text_column=text_column)

        # Evaluate
        metrics = classifier.evaluate(val_df, text_column=text_column)
        metrics['fold'] = fold + 1
        fold_results.append(metrics)

        logger.info(f"Fold {fold+1} Results: {metrics}")

    # Aggregate results
    aggregate = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'precision_std': np.std([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'recall_std': np.std([r['recall'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results]),
        'f1_std': np.std([r['f1'] for r in fold_results]),
        'sensitivity': np.mean([r['sensitivity'] for r in fold_results]),
        'specificity': np.mean([r['specificity'] for r in fold_results]),
    }

    return {
        'fold_results': fold_results,
        'aggregate': aggregate
    }


def hyperparameter_optimization(
    df: pd.DataFrame,
    text_column: str = 'ExaminationText',
    label_column: str = 'Annotation',
    n_trials: int = 20,
    n_folds: int = 3,
    optimization_metric: str = 'f1',
    output_dir: str = './hpo_output',
    use_optuna: bool = True
) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization for BERT classifier.

    Args:
        df: DataFrame with annotated data
        text_column: Column name for text
        label_column: Column name for labels
        n_trials: Number of optimization trials (for Optuna)
        n_folds: Number of CV folds for evaluation
        optimization_metric: Metric to optimize ('f1', 'accuracy', 'sensitivity')
        output_dir: Directory to save results
        use_optuna: If True, use Optuna; if False, use grid search

    Returns:
        Dictionary with best parameters and optimization history
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    df = df[df[label_column].notna()].copy()
    df['label'] = df[label_column].str.strip().str.capitalize()
    df['label_id'] = df['label'].map({'No': 0, 'Yes': 1})

    if use_optuna:
        return _optuna_optimization(
            df, text_column, label_column, n_trials, n_folds,
            optimization_metric, output_dir
        )
    else:
        return _grid_search_optimization(
            df, text_column, label_column, n_folds,
            optimization_metric, output_dir
        )


def _optuna_optimization(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    n_trials: int,
    n_folds: int,
    optimization_metric: str,
    output_dir: str
) -> Dict[str, Any]:
    """Optuna-based Bayesian hyperparameter optimization."""
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        raise ImportError("Optuna not installed. Run: pip install optuna")

    import shutil

    # Store trial history
    trial_history = []

    def objective(trial):
        # Clear GPU memory before each trial (important for MPS on Mac)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc
        gc.collect()

        # Sample hyperparameters - reduced for Mac MPS memory constraints
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
        batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])  # Reduced from [4, 8, 16]
        num_epochs = trial.suggest_int('num_epochs', 2, 5)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.2)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
        max_length = trial.suggest_categorical('max_length', [256, 384, 512])

        # Create config with sampled parameters
        config = ClassifierConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_length=max_length,
            output_dir=f"{output_dir}/trial_{trial.number}",
            early_stopping_patience=2  # Faster stopping for HPO
        )

        # Cross-validation evaluation
        try:
            cv_results = cross_validate(
                df, config,
                text_column=text_column,
                label_column=label_column,
                n_folds=n_folds
            )

            score = cv_results['aggregate'][optimization_metric]

            # Log trial
            trial_info = {
                'trial': trial.number,
                'params': trial.params,
                'score': score,
                'all_metrics': cv_results['aggregate']
            }
            trial_history.append(trial_info)

            logger.info(f"Trial {trial.number}: {optimization_metric}={score:.4f}")
            logger.info(f"  Params: lr={learning_rate:.2e}, batch={batch_size}, "
                       f"epochs={num_epochs}, warmup={warmup_ratio:.2f}")

            # Clean up trial checkpoints to save disk space
            trial_dir = f"{output_dir}/trial_{trial.number}"
            if os.path.exists(trial_dir):
                shutil.rmtree(trial_dir)
                logger.info(f"  Cleaned up trial directory: {trial_dir}")

            # Clear memory after each trial
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return score

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            # Clean up failed trial directory too
            trial_dir = f"{output_dir}/trial_{trial.number}"
            if os.path.exists(trial_dir):
                shutil.rmtree(trial_dir)
            # Clear memory after failed trial
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return 0.0

    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='bert_hpo'
    )

    # Optimize
    logger.info(f"\nStarting Optuna optimization with {n_trials} trials...")
    logger.info(f"Optimizing: {optimization_metric}")
    logger.info(f"Using {n_folds}-fold cross-validation\n")

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Get best results
    best_params = study.best_params
    best_score = study.best_value

    logger.info(f"\n{'='*60}")
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Best {optimization_metric}: {best_score:.4f}")
    logger.info(f"Best parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")

    # Save results
    results = {
        'best_params': best_params,
        'best_score': best_score,
        'optimization_metric': optimization_metric,
        'n_trials': n_trials,
        'n_folds': n_folds,
        'trial_history': trial_history
    }

    with open(f"{output_dir}/optimization_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def _grid_search_optimization(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    n_folds: int,
    optimization_metric: str,
    output_dir: str
) -> Dict[str, Any]:
    """Grid search hyperparameter optimization."""
    from itertools import product
    import shutil
    import gc

    # Define parameter grid - reduced for Mac MPS memory constraints
    param_grid = {
        'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
        'batch_size': [2, 4, 8],  # Reduced from [4, 8, 16]
        'num_epochs': [3, 4, 5],
        # 'warmup_ratio': [0.0, 0.1],
        # 'weight_decay': [0.01, 0.05],
    }

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))

    total_combinations = len(all_combinations)
    logger.info(f"\nStarting Grid Search with {total_combinations} combinations...")
    logger.info(f"Optimizing: {optimization_metric}")
    logger.info(f"Using {n_folds}-fold cross-validation\n")

    results_list = []
    best_score = -1
    best_params = None

    for i, param_combo in enumerate(all_combinations):
        # Clear GPU memory before each combination
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        params = dict(zip(param_names, param_combo))

        logger.info(f"\nCombination {i+1}/{total_combinations}")
        logger.info(f"Params: {params}")

        config = ClassifierConfig(
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            num_epochs=params['num_epochs'],
            # warmup_ratio=params['warmup_ratio'],
            # weight_decay=params['weight_decay'],
            warmup_ratio=0.1,
            weight_decay=0.01,
            output_dir=f"{output_dir}/grid_{i}",
            early_stopping_patience=2
        )

        try:
            cv_results = cross_validate(
                df, config,
                text_column=text_column,
                label_column=label_column,
                n_folds=n_folds
            )

            score = cv_results['aggregate'][optimization_metric]

            # Flatten results for CSV output
            result_entry = {
                'combination': i,
                **params,  # Flatten params (learning_rate, batch_size, num_epochs)
                'f1': cv_results['aggregate']['f1'],
                'accuracy': cv_results['aggregate']['accuracy'],
                'precision': cv_results['aggregate']['precision'],
                'recall': cv_results['aggregate']['recall'],
                'sensitivity': cv_results['aggregate']['sensitivity'],
                'specificity': cv_results['aggregate']['specificity'],
            }
            results_list.append(result_entry)

            if score > best_score:
                best_score = score
                best_params = params.copy()

            logger.info(f"Score: {score:.4f} (best so far: {best_score:.4f})")

            # Clean up grid search checkpoints to save disk space
            grid_dir = f"{output_dir}/grid_{i}"
            if os.path.exists(grid_dir):
                shutil.rmtree(grid_dir)

        except Exception as e:
            logger.warning(f"Combination {i} failed: {e}")
            # Clean up failed combination directory too
            grid_dir = f"{output_dir}/grid_{i}"
            if os.path.exists(grid_dir):
                shutil.rmtree(grid_dir)
            continue

    logger.info(f"\n{'='*60}")
    logger.info("GRID SEARCH COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Best {optimization_metric}: {best_score:.4f}")
    logger.info(f"Best parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")

    # Save results
    results = {
        'best_params': best_params,
        'best_score': best_score,
        'optimization_metric': optimization_metric,
        'total_combinations': total_combinations,
        'n_folds': n_folds,
        'all_results': results_list
    }

    with open(f"{output_dir}/grid_search_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save as DataFrame for easy viewing
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f"{output_dir}/grid_search_results.csv", index=False)

    return results


def quick_hyperparameter_search(
    df: pd.DataFrame,
    text_column: str = 'ExaminationText',
    label_column: str = 'Annotation',
    output_dir: str = './hpo_output'
) -> Dict[str, Any]:
    """
    Quick hyperparameter search focusing on most impactful parameters.

    Tests learning rates and batch sizes only (fastest search).

    Args:
        df: DataFrame with annotated data
        text_column: Column name for text
        label_column: Column name for labels
        output_dir: Directory to save results

    Returns:
        Dictionary with best parameters
    """
    import shutil
    import gc
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    df = df[df[label_column].notna()].copy()
    df['label'] = df[label_column].str.strip().str.capitalize()
    df['label_id'] = df['label'].map({'No': 0, 'Yes': 1})

    # Quick grid: most impactful parameters only - reduced for Mac MPS
    learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
    batch_sizes = [4, 8]  # Reduced from [8, 16]

    results = []
    best_f1 = -1
    best_config = None

    total = len(learning_rates) * len(batch_sizes)
    current = 0

    logger.info(f"\nQuick HPO: Testing {total} configurations...")

    for lr in learning_rates:
        for batch_size in batch_sizes:
            # Clear GPU memory before each config
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            current += 1
            logger.info(f"\n[{current}/{total}] lr={lr:.0e}, batch_size={batch_size}")

            config = ClassifierConfig(
                learning_rate=lr,
                batch_size=batch_size,
                num_epochs=3,  # Fixed for speed
                output_dir=f"{output_dir}/quick_{current}",
                early_stopping_patience=2
            )

            try:
                # Use 3-fold CV for speed
                cv_results = cross_validate(
                    df, config,
                    text_column=text_column,
                    label_column=label_column,
                    n_folds=3
                )

                f1 = cv_results['aggregate']['f1']
                acc = cv_results['aggregate']['accuracy']

                results.append({
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'f1': f1,
                    'accuracy': acc,
                    'precision': cv_results['aggregate']['precision'],
                    'recall': cv_results['aggregate']['recall'],
                    'sensitivity': cv_results['aggregate']['sensitivity'],
                    'specificity': cv_results['aggregate']['specificity'],
                })

                logger.info(f"  F1: {f1:.4f}, Accuracy: {acc:.4f}, Sensitivity: {cv_results['aggregate']['sensitivity']:.4f}, Specificity: {cv_results['aggregate']['specificity']:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_config = {'learning_rate': lr, 'batch_size': batch_size}

                # Clean up quick HPO checkpoints to save disk space
                quick_dir = f"{output_dir}/quick_{current}"
                if os.path.exists(quick_dir):
                    shutil.rmtree(quick_dir)

            except Exception as e:
                logger.warning(f"  Failed: {e}")
                # Clean up failed directory too
                quick_dir = f"{output_dir}/quick_{current}"
                if os.path.exists(quick_dir):
                    shutil.rmtree(quick_dir)
                continue

    logger.info(f"\n{'='*60}")
    logger.info("QUICK HPO COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Best F1: {best_f1:.4f}")
    logger.info(f"Best config: lr={best_config['learning_rate']:.0e}, "
               f"batch_size={best_config['batch_size']}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/quick_hpo_results.csv", index=False)

    return {
        'best_params': best_config,
        'best_f1': best_f1,
        'all_results': results
    }


def train_with_best_params(
    df: pd.DataFrame,
    best_params: Dict[str, Any],
    text_column: str = 'ExaminationText',
    label_column: str = 'Annotation',
    output_dir: str = './best_model'
) -> Tuple[BERTNutritionClassifier, Dict[str, float]]:
    """
    Train final model using best hyperparameters.

    Args:
        df: DataFrame with annotated data
        best_params: Best parameters from optimization
        text_column: Column name for text
        label_column: Column name for labels
        output_dir: Directory to save final model

    Returns:
        Trained classifier and test metrics
    """
    logger.info(f"\nTraining final model with best parameters...")
    logger.info(f"Parameters: {best_params}")

    config = ClassifierConfig(
        learning_rate=best_params.get('learning_rate', 2e-5),
        batch_size=best_params.get('batch_size', 8),
        num_epochs=best_params.get('num_epochs', 5),
        warmup_ratio=best_params.get('warmup_ratio', 0.1),
        weight_decay=best_params.get('weight_decay', 0.01),
        max_length=best_params.get('max_length', 512),
        output_dir=output_dir
    )

    classifier = BERTNutritionClassifier(config)
    train_df, test_df = classifier.prepare_data(
        df, text_column=text_column, label_column=label_column
    )

    classifier.load_model()
    classifier.train(train_df, test_df, text_column=text_column)

    # Evaluate on test set
    metrics = classifier.evaluate(test_df, text_column=text_column)

    # Save final model
    classifier.save_model(f"{output_dir}/final_model")

    logger.info(f"\nFinal model saved to {output_dir}/final_model")
    logger.info(f"Test metrics: {metrics}")

    return classifier, metrics


# def compare_models(
#     df: pd.DataFrame,
#     models: List[str] = ['bio_clinical_bert', 'vet_bert'],
#     text_column: str = 'ExaminationText',
#     label_column: str = 'Annotation',
#     test_size: float = 0.2
# ) -> pd.DataFrame:
#     """
#     Compare multiple models on the same data.

#     Args:
#         df: DataFrame with data
#         models: List of model names to compare
#         text_column: Column name for text
#         label_column: Column name for labels
#         test_size: Test set size

#     Returns:
#         DataFrame with comparison results
#     """
#     results = []

#     for model_name in models:
#         logger.info(f"\n{'='*60}")
#         logger.info(f"Training: {model_name}")
#         logger.info(f"{'='*60}")

#         config = ClassifierConfig(
#             model_name=model_name,
#             output_dir=f'./output/{model_name}'
#         )

#         classifier = BERTNutritionClassifier(config)
#         train_df, test_df = classifier.prepare_data(
#             df, text_column=text_column, label_column=label_column, test_size=test_size
#         )

#         classifier.load_model()
#         classifier.train(train_df, test_df, text_column=text_column)

#         metrics = classifier.evaluate(test_df, text_column=text_column)
#         metrics['model'] = model_name

#         results.append(metrics)

#         logger.info(f"{model_name} Results: {metrics}")

#     return pd.DataFrame(results)


# def print_comparison_results(results_df: pd.DataFrame):
#     """Print comparison results in a formatted table."""
#     print("\n" + "=" * 80)
#     print("MODEL COMPARISON RESULTS")
#     print("=" * 80)

#     # Format for display
#     display_cols = ['model', 'accuracy', 'precision', 'recall', 'f1', 'sensitivity', 'specificity']
#     display_df = results_df[display_cols].copy()

#     for col in display_cols[1:]:
#         display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

#     print(display_df.to_string(index=False))
#     print("=" * 80)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='BERT-based nutrition classifier')
    parser.add_argument('--cv', type=int, default=0, help='Number of CV folds (0=no CV)')
    parser.add_argument('--hpo', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--hpo-trials', type=int, default=20, help='Number of HPO trials (Optuna)')
    parser.add_argument('--hpo-method', choices=['optuna', 'grid', 'quick'], default='optuna',
                       help='HPO method: optuna (Bayesian), grid (exhaustive), quick (fast)')
    parser.add_argument('--input', type=str, default='../scripts/output/strat_sample_v1_cleaned_df.xlsx',
                       help='Input data file')
    parser.add_argument('--text-col', type=str, default='cleaned_examination_text',
                       help='Text column name')
    parser.add_argument('--label-col', type=str, default='Annotation',
                       help='Label column name')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory')
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.input}")
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input)
    logger.info(f"Loaded {len(df)} records")

    if args.hpo:
        # Hyperparameter optimization
        logger.info(f"\n{'='*60}")
        logger.info("HYPERPARAMETER OPTIMIZATION")
        logger.info(f"{'='*60}")

        if args.hpo_method == 'quick':
            results = quick_hyperparameter_search(
                df,
                text_column=args.text_col,
                label_column=args.label_col,
                output_dir=f"{args.output_dir}/hpo_quick"
            )
        elif args.hpo_method == 'grid':
            results = hyperparameter_optimization(
                df,
                text_column=args.text_col,
                label_column=args.label_col,
                n_folds=3,
                output_dir=f"{args.output_dir}/hpo_grid",
                use_optuna=False
            )
        else:  # optuna
            results = hyperparameter_optimization(
                df,
                text_column=args.text_col,
                label_column=args.label_col,
                n_trials=args.hpo_trials,
                n_folds=3,
                output_dir=f"{args.output_dir}/hpo_optuna",
                use_optuna=True
            )

        # Train final model with best params
        logger.info("\nTraining final model with best parameters...")
        classifier, metrics = train_with_best_params(
            df,
            results['best_params'],
            text_column=args.text_col,
            label_column=args.label_col,
            output_dir=f"{args.output_dir}/best_model"
        )

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Best parameters: {results['best_params']}")
        print(f"Test metrics: {metrics}")

    elif args.cv > 0:
        # Cross-validation
        config = ClassifierConfig(output_dir=args.output_dir)
        cv_results = cross_validate(
            df, config,
            text_column=args.text_col,
            label_column=args.label_col,
            n_folds=args.cv
        )
        logger.info(f"\nAggregate CV Results: {cv_results['aggregate']}")

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f'{args.output_dir}/cv_results.json', 'w') as f:
            json.dump(cv_results, f, indent=2)

    else:
        # Single training run
        config = ClassifierConfig(output_dir=args.output_dir)

        classifier = BERTNutritionClassifier(config)
        train_df, test_df = classifier.prepare_data(
            df,
            text_column=args.text_col,
            label_column=args.label_col
        )

        classifier.load_model()
        classifier.train(train_df, test_df, text_column=args.text_col)

        metrics = classifier.evaluate(test_df, text_column=args.text_col)

        print("\n" + "=" * 60)
        print("FINAL TEST RESULTS")
        print("=" * 60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f} ({value*100:.2f}%)")
            else:
                print(f"{key}: {value}")
        print("=" * 60)

        # Save model
        classifier.save_model('./final_model')

    # ============================================================
    # Cross Validation 
    # ============================================================
    # config = ClassifierConfig()
    # cv_results = cross_validate(
    #     df, config,
    #     text_column="cleaned_examination_text",
    #     label_column="Annotation",
    #     n_folds=5
    # )
    # logger.info(f"\nAggregate CV Results: {cv_results['aggregate']}")

    # # Save results
    # with open('./output/cv_results.json', 'w') as f:
    #     json.dump(cv_results, f, indent=2)

    # parser = argparse.ArgumentParser(description='BERT-based nutrition classifier')
    # parser.add_argument('--input', '-i', required=True, help='Input file (CSV or Excel)')
    # parser.add_argument('--text-column', default='ExaminationText', help='Text column name')
    # parser.add_argument('--label-column', default='Annotation', help='Label column name')
    # parser.add_argument('--model', default='bio_clinical_bert',
    #                    choices=list(AVAILABLE_MODELS.keys()),
    #                    help='Model to use')
    # parser.add_argument('--output-dir', default='./bert_output', help='Output directory')
    # parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    # parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    # parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    # parser.add_argument('--compare', action='store_true', help='Compare all models')
    # parser.add_argument('--cv', type=int, default=0, help='Number of CV folds (0=no CV)')

    # args = parser.parse_args()

    # # Load data
    # if args.input.endswith('.csv'):
    #     df = pd.read_csv(args.input)
    # else:
    #     df = pd.read_excel(args.input)

    # logger.info(f"Loaded {len(df)} records")

    # if args.compare:
    #     # Compare models
    #     results = compare_models(
    #         df,
    #         models=['bio_clinical_bert', 'vet_bert'],
    #         text_column=args.text_column,
    #         label_column=args.label_column
    #     )
    #     print_comparison_results(results)
    #     results.to_csv(f'{args.output_dir}/model_comparison.csv', index=False)

    # elif args.cv > 0:
    #     # Cross-validation
    #     config = ClassifierConfig(
    #         model_name=args.model,
    #         num_epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         learning_rate=args.learning_rate,
    #         output_dir=args.output_dir
    #     )
    #     cv_results = cross_validate(
    #         df, config,
    #         text_column=args.text_column,
    #         label_column=args.label_column,
    #         n_folds=args.cv
    #     )
    #     logger.info(f"\nAggregate CV Results: {cv_results['aggregate']}")

    #     # Save results
    #     with open(f'{args.output_dir}/cv_results.json', 'w') as f:
    #         json.dump(cv_results, f, indent=2)

    # else:
    #     # Single training run
    #     config = ClassifierConfig(
    #         model_name=args.model,
    #         num_epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         learning_rate=args.learning_rate,
    #         output_dir=args.output_dir
    #     )

    #     classifier = BERTNutritionClassifier(config)
    #     train_df, test_df = classifier.prepare_data(
    #         df,
    #         text_column=args.text_column,
    #         label_column=args.label_column
    #     )

    #     classifier.load_model()
    #     classifier.train(train_df, test_df, text_column=args.text_column)

    #     metrics = classifier.evaluate(test_df, text_column=args.text_column)

    #     print("\n" + "=" * 60)
    #     print("FINAL TEST RESULTS")
    #     print("=" * 60)
    #     for key, value in metrics.items():
    #         if isinstance(value, float):
    #             print(f"{key}: {value:.4f} ({value*100:.2f}%)")
    #         else:
    #             print(f"{key}: {value}")
    #     print("=" * 60)

    #     # Save model
    #     classifier.save_model(f'{args.output_dir}/final_model')


# ============================================================
# FINAL TEST RESULTS
# ============================================================
# accuracy: 0.9154 (91.54%)
# precision: 0.8710 (87.10%)
# recall: 0.9419 (94.19%)
# f1: 0.9050 (90.50%)
# sensitivity: 0.9419 (94.19%)
# specificity: 0.8957 (89.57%)
# true_positives: 81
# true_negatives: 103
# false_positives: 12
# false_negatives: 5