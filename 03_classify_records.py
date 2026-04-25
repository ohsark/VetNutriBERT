"""
Classify Records and Generate Demographics
============================================

This script loads a DataFrame, classifies records using the trained BERT model,
and generates basic demographics about the classified records.

Usage:
    python 03_classify_records.py --input path/to/data.xlsx --output path/to/output/
"""

import argparse
import logging
import os
from typing import Dict, Any

import pandas as pd

# Import the BERT classifier from the module file
# Note: The module is named with dots, so we use importlib
import importlib.util
import sys

def _import_bert_classifier():
    """Import the BERT classifier module with non-standard filename."""
    module_path = os.path.join(os.path.dirname(__file__), '02.2_bert_classifier.py')
    spec = importlib.util.spec_from_file_location("bert_classifier", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bert_classifier"] = module
    spec.loader.exec_module(module)
    return module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(input_path: str) -> pd.DataFrame:
    """Load data from CSV or Excel file."""
    logger.info(f"Loading data from {input_path}")

    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    logger.info(f"Loaded {len(df)} records")
    return df


def classify_records(
    df: pd.DataFrame,
    model_path: str = './final_model',
    text_column: str = 'cleaned_examination_text',
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Classify records using the trained BERT model.

    Args:
        df: DataFrame with text data
        model_path: Path to the trained model
        text_column: Column containing text to classify
        batch_size: Batch size for inference

    Returns:
        DataFrame with added prediction columns
    """
    logger.info(f"Loading trained model from {model_path}")

    # Import and initialize classifier
    bert_module = _import_bert_classifier()
    classifier = bert_module.BERTNutritionClassifier()
    classifier.load_trained_model(model_path)

    # Get texts for classification
    texts = df[text_column].fillna('').tolist()

    logger.info(f"Classifying {len(texts)} records...")
    predictions, probabilities = classifier.predict(texts, batch_size=batch_size)

    # Add predictions to DataFrame
    df = df.copy()
    df['predicted_nutrition'] = predictions
    df['nutrition_probability'] = probabilities

    # Summary
    yes_count = (df['predicted_nutrition'] == 'Yes').sum()
    no_count = (df['predicted_nutrition'] == 'No').sum()
    logger.info(f"Classification complete: {yes_count} Yes, {no_count} No")

    return df


def generate_demographics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate basic demographics about the records.

    Args:
        df: DataFrame with records (after classification)

    Returns:
        Dictionary containing demographic statistics
    """
    demographics = {}

    # Total records
    demographics['total_records'] = len(df)

    # Classification distribution (if available)
    if 'predicted_nutrition' in df.columns:
        nutrition_dist = df['predicted_nutrition'].value_counts().to_dict()
        demographics['nutrition_classification'] = nutrition_dist
        demographics['nutrition_positive_rate'] = (
            nutrition_dist.get('Yes', 0) / len(df) * 100
        )

    # Year distribution
    if 'YearConsult' in df.columns:
        demographics['year_distribution'] = df['YearConsult'].value_counts().sort_index().to_dict()

    # Sex distribution
    if 'PatientSex' in df.columns:
        demographics['sex_distribution'] = df['PatientSex'].value_counts().to_dict()

    # Breed distribution (top 10)
    if 'PatientBreed' in df.columns:
        breed_counts = df['PatientBreed'].value_counts()
        demographics['breed_distribution_top10'] = breed_counts.head(10).to_dict()
        demographics['unique_breeds'] = df['PatientBreed'].nunique()

    # Breed size distribution
    if 'BreedSize' in df.columns:
        demographics['breed_size_distribution'] = df['BreedSize'].value_counts().to_dict()

    # Age statistics
    if 'AgeYears' in df.columns:
        demographics['age_statistics'] = {
            'mean': round(df['AgeYears'].mean(), 2),
            'median': round(df['AgeYears'].median(), 2),
            'min': int(df['AgeYears'].min()),
            'max': int(df['AgeYears'].max()),
            'std': round(df['AgeYears'].std(), 2)
        }

    # Age group distribution
    if 'AgeGroup' in df.columns:
        demographics['age_group_distribution'] = df['AgeGroup'].value_counts().to_dict()

    # Database/clinic distribution
    if 'DatabaseName' in df.columns:
        demographics['database_distribution'] = df['DatabaseName'].value_counts().to_dict()

    if 'ClinicCodeConsult' in df.columns:
        demographics['unique_clinics'] = df['ClinicCodeConsult'].nunique()

    # Postcode distribution (unique count)
    if 'Postcode' in df.columns:
        demographics['unique_postcodes'] = df['Postcode'].nunique()

    return demographics


def print_demographics(demographics: Dict[str, Any]) -> None:
    """Print demographics in a formatted way."""
    print("\n" + "=" * 70)
    print("DEMOGRAPHICS SUMMARY")
    print("=" * 70)

    print(f"\nTotal Records: {demographics['total_records']:,}")

    if 'nutrition_classification' in demographics:
        print(f"\n--- Nutrition Classification ---")
        for label, count in demographics['nutrition_classification'].items():
            pct = count / demographics['total_records'] * 100
            print(f"  {label}: {count:,} ({pct:.1f}%)")

    if 'sex_distribution' in demographics:
        print(f"\n--- Sex Distribution ---")
        for sex, count in demographics['sex_distribution'].items():
            pct = count / demographics['total_records'] * 100
            print(f"  {sex}: {count:,} ({pct:.1f}%)")

    if 'breed_size_distribution' in demographics:
        print(f"\n--- Breed Size Distribution ---")
        for size, count in demographics['breed_size_distribution'].items():
            pct = count / demographics['total_records'] * 100
            print(f"  {size}: {count:,} ({pct:.1f}%)")

    if 'age_statistics' in demographics:
        print(f"\n--- Age Statistics (Years) ---")
        stats = demographics['age_statistics']
        print(f"  Mean: {stats['mean']}")
        print(f"  Median: {stats['median']}")
        print(f"  Range: {stats['min']} - {stats['max']}")
        print(f"  Std Dev: {stats['std']}")

    if 'age_group_distribution' in demographics:
        print(f"\n--- Age Group Distribution ---")
        for group, count in demographics['age_group_distribution'].items():
            pct = count / demographics['total_records'] * 100
            print(f"  {group}: {count:,} ({pct:.1f}%)")

    if 'breed_distribution_top10' in demographics:
        print(f"\n--- Top 10 Breeds ---")
        for breed, count in demographics['breed_distribution_top10'].items():
            pct = count / demographics['total_records'] * 100
            print(f"  {breed}: {count:,} ({pct:.1f}%)")
        if 'unique_breeds' in demographics:
            print(f"  ... ({demographics['unique_breeds']} unique breeds total)")

    if 'year_distribution' in demographics:
        print(f"\n--- Year Distribution ---")
        for year, count in demographics['year_distribution'].items():
            pct = count / demographics['total_records'] * 100
            print(f"  {year}: {count:,} ({pct:.1f}%)")

    if 'database_distribution' in demographics:
        print(f"\n--- Database Distribution ---")
        for db, count in demographics['database_distribution'].items():
            pct = count / demographics['total_records'] * 100
            print(f"  {db}: {count:,} ({pct:.1f}%)")

    if 'unique_clinics' in demographics:
        print(f"\nUnique Clinics: {demographics['unique_clinics']}")

    if 'unique_postcodes' in demographics:
        print(f"Unique Postcodes: {demographics['unique_postcodes']}")

    print("\n" + "=" * 70)


def save_results(
    df: pd.DataFrame,
    demographics: Dict[str, Any],
    output_dir: str
) -> None:
    """Save classified data and demographics to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Save classified DataFrame
    output_file = os.path.join(output_dir, 'classified_records.xlsx')
    df.to_excel(output_file, index=False)
    logger.info(f"Saved classified records to {output_file}")

    # Save demographics as JSON
    import json
    demographics_file = os.path.join(output_dir, 'demographics.json')
    with open(demographics_file, 'w') as f:
        json.dump(demographics, f, indent=2, default=str)
    logger.info(f"Saved demographics to {demographics_file}")

    # Save demographics summary as text
    summary_file = os.path.join(output_dir, 'demographics_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("DEMOGRAPHICS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Records: {demographics['total_records']:,}\n\n")

        if 'nutrition_classification' in demographics:
            f.write("Nutrition Classification:\n")
            for label, count in demographics['nutrition_classification'].items():
                pct = count / demographics['total_records'] * 100
                f.write(f"  {label}: {count:,} ({pct:.1f}%)\n")
            f.write("\n")

        if 'sex_distribution' in demographics:
            f.write("Sex Distribution:\n")
            for sex, count in demographics['sex_distribution'].items():
                pct = count / demographics['total_records'] * 100
                f.write(f"  {sex}: {count:,} ({pct:.1f}%)\n")
            f.write("\n")

        if 'breed_size_distribution' in demographics:
            f.write("Breed Size Distribution:\n")
            for size, count in demographics['breed_size_distribution'].items():
                pct = count / demographics['total_records'] * 100
                f.write(f"  {size}: {count:,} ({pct:.1f}%)\n")
            f.write("\n")

        if 'age_statistics' in demographics:
            f.write("Age Statistics (Years):\n")
            stats = demographics['age_statistics']
            f.write(f"  Mean: {stats['mean']}\n")
            f.write(f"  Median: {stats['median']}\n")
            f.write(f"  Range: {stats['min']} - {stats['max']}\n")
            f.write(f"  Std Dev: {stats['std']}\n")
            f.write("\n")

        if 'age_group_distribution' in demographics:
            f.write("Age Group Distribution:\n")
            for group, count in demographics['age_group_distribution'].items():
                pct = count / demographics['total_records'] * 100
                f.write(f"  {group}: {count:,} ({pct:.1f}%)\n")
            f.write("\n")

        if 'breed_distribution_top10' in demographics:
            f.write("Top 10 Breeds:\n")
            for breed, count in demographics['breed_distribution_top10'].items():
                pct = count / demographics['total_records'] * 100
                f.write(f"  {breed}: {count:,} ({pct:.1f}%)\n")
            if 'unique_breeds' in demographics:
                f.write(f"  ({demographics['unique_breeds']} unique breeds total)\n")
            f.write("\n")

    logger.info(f"Saved demographics summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Classify records using BERT and generate demographics'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input data file (CSV or Excel)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./output/classified',
        help='Output directory for results'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='./final_model',
        help='Path to trained BERT model'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='cleaned_examination_text',
        help='Column containing text to classify'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--skip-classification',
        action='store_true',
        help='Skip classification and only generate demographics'
    )

    args = parser.parse_args()

    # Load data
    df = load_data(args.input)

    # Classify records (unless skipped)
    if not args.skip_classification:
        df = classify_records(
            df,
            model_path=args.model_path,
            text_column=args.text_column,
            batch_size=args.batch_size
        )

    # Generate demographics
    demographics = generate_demographics(df)

    # Print demographics
    print_demographics(demographics)

    # Save results
    save_results(df, demographics, args.output)

    logger.info("Done!")


if __name__ == "__main__":
    main()
