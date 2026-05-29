import pandas as pd
import numpy as np

print("=" * 50)
print("FAKE NEWS DETECTION SYSTEM")
print("=" * 50)

# Load the datasets
print("\n📂 Loading datasets...")
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')
print("✅ Datasets loaded successfully!")

# Check dataset sizes
print(f"\n📊 Dataset Info:")
print(f"   Fake news articles: {len(fake_df)}")
print(f"   Real news articles: {len(true_df)}")

# Add labels
fake_df['label'] = 0  # 0 = Fake
true_df['label'] = 1  # 1 = Real

# Create combined text column (title + text) - THIS WAS MISSING
fake_df['full_text'] = fake_df['title'] + " " + fake_df['text']
true_df['full_text'] = true_df['title'] + " " + true_df['text']

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n📈 Combined dataset: {len(df)} total articles")
print(f"   Class distribution:")
print(f"   Fake (0): {len(df[df['label']==0])}")
print(f"   Real (1): {len(df[df['label']==1])}")

# Show sample
print("\n📝 First 2 rows:")
print(df[['title', 'label']].head(2))

# Verify columns exist
print(f"\n📋 Columns in dataset: {df.columns.tolist()}")

# Save processed data with full_text column
df.to_csv('data/processed_news.csv', index=False)
print("\n💾 Saved processed data to 'data/processed_news.csv'")

print("\n✅ Setup complete! Ready for next step.")