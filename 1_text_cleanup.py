import pandas as pd
import re
import string
from spellchecker import SpellChecker
import time
import multiprocessing as mp
import os


def parallel_text_processor_single(args):
    """
    Top-level function for parallel text processing - processes a SINGLE text
    This function must be at module level to be pickable for multiprocessing
    """
    text, synonym_file_path = args
    
    # Each worker loads its own data on first call using function attributes
    # This avoids pickling issues while still being efficient
    if not hasattr(parallel_text_processor_single, 'initialized'):
        try:
            synonyms_df = pd.read_excel(synonym_file_path)
            parallel_text_processor_single.spell = SpellChecker()
            parallel_text_processor_single.spell.word_frequency.load_words(
                set(synonyms_df["Synonym"].dropna().str.lower().tolist()) |
                set(synonyms_df["Word"].dropna().str.lower().tolist())
            )
            parallel_text_processor_single.synonym_dict = dict(
                zip(synonyms_df['Word'].str.lower(), synonyms_df['Synonym'].str.lower())
            )
            parallel_text_processor_single.spell_cache = {}
            parallel_text_processor_single.initialized = True
        except Exception as e:
            # Fallback if synonyms can't be loaded
            parallel_text_processor_single.spell = SpellChecker()
            parallel_text_processor_single.synonym_dict = {}
            parallel_text_processor_single.spell_cache = {}
            parallel_text_processor_single.initialized = True
    
    def local_spellcheck(text):
        if pd.isna(text):
            return text
        
        text = text.lstrip(string.punctuation)
        tokens = re.findall(r'\b\w+\b|\W+', text)
        corrected_tokens = []
            
        for token in tokens:
            if re.match(r'\b\w+\b', token):
                word_lower = token.lower()
                
                if len(word_lower) <= 2 or re.search(r'\d', word_lower):
                    corrected_tokens.append(token)
                    continue
                
                if word_lower in parallel_text_processor_single.spell_cache:
                    correction = parallel_text_processor_single.spell_cache[word_lower]
                else:
                    if word_lower not in parallel_text_processor_single.spell:
                        correction = parallel_text_processor_single.spell.correction(word_lower)
                        parallel_text_processor_single.spell_cache[word_lower] = correction
                    else:
                        correction = word_lower
                        parallel_text_processor_single.spell_cache[word_lower] = word_lower
                
                if correction and correction != word_lower:
                    if token.isupper():
                        corrected_tokens.append(correction.upper())
                    elif token.istitle():
                        corrected_tokens.append(correction.capitalize())
                    else:
                        corrected_tokens.append(correction)
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)
        
        return ''.join(corrected_tokens)
    
    # Process the single text
    if pd.isna(text):
        return text
    
    text = local_spellcheck(text)
    cleaned = text.strip().lower()
    
    for word, synonym in parallel_text_processor_single.synonym_dict.items():
        cleaned = re.sub(rf"\b{re.escape(word)}\b", synonym, cleaned)

    # cleaned = re.sub(rf"[{re.escape(string.punctuation.replace('.', '').replace(',', '').replace(':', ''))}]", " ", cleaned)
    cleaned = re.sub(rf"[{re.escape(string.punctuation)}]", " ", cleaned)
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)  # Remove unicode characters
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def clean_dataframe(df, text_column='ExaminationText', output_file=None, n_workers=None):
    """
    Clean text in dataframe using parallel processing with Pool.imap

    Args:
        df: DataFrame containing the text data
        text_column: Name of the column to clean
        output_file: Optional output file path
        n_workers: Number of worker processes (default: 6)

    Returns:
        DataFrame with cleaned text column
    """
    if n_workers is None:
        n_workers = 6  # Cap at 8 to avoid overwhelming system
    
    print(f"\nProcessing {len(df):,} records using {n_workers} workers (TRUE PARALLEL)...")
    print("=" * 70)
    
    start_time = time.time()
    
    # Prepare arguments for each text
    synonym_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data', 'Lyka_Synonyms.xlsx')
    args_list = [(text, synonym_file_path) for text in df[text_column]]
    
    # Calculate optimal chunksize
    chunksize = max(1, len(df) // (n_workers * 4))  # 4 chunks per worker
    print(f"Using chunksize: {chunksize}")
    
    # Process in parallel using Pool.imap
    cleaned_texts = []
    try:
        with mp.Pool(processes=n_workers) as pool:
            total_processed = 0
            for result in pool.imap(parallel_text_processor_single, args_list, chunksize=chunksize):
                cleaned_texts.append(result)
                total_processed += 1
                
                # Progress update every 100 records
                if total_processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    remaining = (len(df) - total_processed) / rate if rate > 0 else 0
                    print(f"Progress: {total_processed:,}/{len(df):,} ({total_processed/len(df)*100:.1f}%) "
                          f"- {rate:.1f} rec/sec - ETA: {remaining:.0f}s")
    
    except Exception as e:
        print(f"Parallel processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    df['cleaned_examination_text'] = cleaned_texts
    
    elapsed_time = time.time() - start_time
    records_per_sec = len(df) / elapsed_time
    
    print("=" * 70)
    print(f"✅ Processed {len(df):,} records in {elapsed_time:.2f} seconds")
    print(f"✅ Speed: {records_per_sec:.1f} records/second")
    print(f"✅ Used {n_workers} parallel workers")
    print("=" * 70)
    
    # Save if output file specified
    if output_file:
        df.to_excel(output_file, index=False)
        print(f"\n✅ Saved to {output_file}")
    
    return df

def initiate_processing(data_name, data_path):
    print("\n" + "=" * 70)
    print("Processing Dataset - ", data_name)
    print("=" * 70)

    # Load the actual data
    if(data_path.endswith('.xlsx') or data_path.endswith('.xls')):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    # Show processing options
    print(f"Available CPU cores: {mp.cpu_count()}")
    print(f"Dataset size: {len(df):,} records")

    # Process with parallel processing
    print("\n" + "=" * 70)
    print("PARALLEL PROCESSING:")
    print("=" * 70)
    start_time = time.time()
    df = clean_dataframe(df, text_column='ExaminationText', output_file=f'./processed_data/{data_name}_cleaned.xlsx')
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE:")
    print("=" * 80)
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Total records: {len(df):,}")
    print(f"Average speed: {len(df)/elapsed:.1f} records/second")
    print("=" * 80)


if __name__ == "__main__":
    # Required for multiprocessing on macOS/Windows
    mp.set_start_method('spawn', force=True)

    # Test with examples using parallel processor
    test_examples = [
        "...preseted with diarrea and v+ seen",
        "!!!Examinaton was comfotable, no V+ or D+",
        "---recomended ab therapy for v+",
        "***palpaton of abdo, client reports v+ episodes"
    ]

    synonym_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data', 'Lyka_Synonyms.xlsx')

    print("Spell check and cleaning examples:")
    print("=" * 70)
    for example in test_examples:
        cleaned = parallel_text_processor_single((example, synonym_file_path))
        print(f"Original: {example}")
        print(f"Cleaned:  {cleaned}")
        print("-" * 70)

    # print(f"{'=' * 70}\nSTRAT SAMPLE V1 PROCESSING\n{'=' * 70}")
    # initiate_processing('strat_sample_v1', '../Data/validation/strat_sample_v1_annotated.xlsx')

    print(f"{'=' * 70}\nDATASET B PROCESSING\n{'=' * 70}")
    initiate_processing('Dataset_B', '../Data/Dataset_B.csv')

    # print(f"{'=' * 70}\nDATASET B PROCESSING\n{'=' * 70}")
    # initiate_processing('Dataset_C', '../Data/Dataset_C.csv')

    # Show sample results
    # print("\nSample results:")
    # for i in range(min(3, len(df))):
    #     if pd.notna(df.iloc[i]['ExaminationText']):
    #         print(f"\nRecord {i+1}:")
    #         print(f"Original (first 100 chars): {df.iloc[i]['ExaminationText'][:100]}...")
    #         print(f"Cleaned (first 100 chars):  {df.iloc[i]['cleaned_examination_text'][:100]}...")

