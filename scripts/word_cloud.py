import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from typing import Optional, Dict
import re
import os

def clean_text(text: str) -> str:
    """Clean text by removing special characters and extra spaces."""
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

def create_prompt_wordcloud(
    annotation_file: str,
    output_path: Optional[str] = None,
    min_freq: int = 5,
    width: int = 800,
    height: int = 400,
    background_color: str = 'white'
) -> None:
    """Create word cloud from dataset prompts."""
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Collect all prompt components
    all_words = []
    for model_info in annotations.values():
        # Add name
        name = model_info.get('name', '')
        if name:
            all_words.extend(clean_text(name).split())
        
        # Add categories
        categories = model_info.get('categories', [])
        for cat in categories:
            if isinstance(cat, dict) and 'name' in cat:
                all_words.extend(clean_text(cat['name']).split())
        
        # Add tags
        tags = model_info.get('tags', [])
        for tag in tags:
            if isinstance(tag, dict) and 'name' in tag:
                all_words.extend(clean_text(tag['name']).split())
    
    # Count word frequencies
    word_freq = Counter(all_words)
    
    # Filter by minimum frequency
    word_freq = {word: count for word, count in word_freq.items() 
                if count >= min_freq}
    
    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        collocations=False,
        min_font_size=10,
        max_font_size=100
    ).generate_from_frequencies(word_freq)
    
    # Plot
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Word cloud saved to: {output_path}")
    else:
        plt.show()

def get_prompt_statistics(annotation_file: str) -> Dict:
    """Get statistics about prompts in the dataset."""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
        
    total_prompts = len(annotations)
    words_per_prompt = []
    category_counts = Counter()
    tag_counts = Counter()
    
    for model_info in annotations.values():
        name = model_info.get('name', '')
        if name:
            words_per_prompt.append(len(clean_text(name).split()))
        
        categories = model_info.get('categories', [])
        for cat in categories:
            if isinstance(cat, dict) and 'name' in cat:
                category_counts[clean_text(cat['name'])] += 1
                
        tags = model_info.get('tags', [])
        for tag in tags:
            if isinstance(tag, dict) and 'name' in tag:
                tag_counts[clean_text(tag['name'])] += 1
                
    return {
        'total_prompts': total_prompts,
        'avg_words_per_prompt': np.mean(words_per_prompt),
        'top_categories': dict(category_counts.most_common(10)),
        'top_tags': dict(tag_counts.most_common(10))
    }

def main():
    # Define paths
    base_dir = Path(os.getcwd())
    annotation_file = base_dir / "objaverse_data/annotations.json"
    plots_dir = base_dir / "plots"
    
    # Create plots directory
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate word cloud
    print("Generating word cloud...")
    create_prompt_wordcloud(
        annotation_file=str(annotation_file),
        output_path=str(plots_dir / "prompt_wordcloud.png"),
        min_freq=10,
        width=1200,
        height=800
    )
    
    # Get and print prompt statistics
    print("\nCalculating prompt statistics...")
    stats = get_prompt_statistics(str(annotation_file))
    
    print(f"\nDataset Statistics:")
    print(f"Total prompts: {stats['total_prompts']}")
    print(f"Average words per prompt: {stats['avg_words_per_prompt']:.1f}")
    
    print("\nTop 10 Categories:")
    for cat, count in stats['top_categories'].items():
        print(f"  {cat}: {count}")
        
    print("\nTop 10 Tags:")
    for tag, count in stats['top_tags'].items():
        print(f"  {tag}: {count}")

if __name__ == "__main__":
    main()