import requests
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm

def fetch_puthiyathalaimurai_api(category, offset=0, limit=20):
    """
    Fetch headlines and categories from the official API endpoint
    """
    url = f"https://www.puthiyathalaimurai.com/api/v1/collections/{category}?item-type=story&offset={offset}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    headlines = []
    items = data.get("items", [])
    
    for item in items:
        story = item.get("story", {})
        title = story.get("headline", "")
        sections = story.get("sections", [])
        url = story.get("url", "")
        
        # Decode Tamil unicode escapes like \u0BAA\u0BBF... → readable Tamil
        title_tamil = title
        
        # Get category (display-name)
        category = None
        if sections and isinstance(sections, list):
            category = sections[0].get("display-name", "general")
        
        headlines.append({
            "title": title_tamil.strip(),
            "url": url if url else "",
            "category": category
        })
        

    return headlines


# Fetch multiple pages (optional)
all_data = []
categories = ['tamilnadu', 'cinema', 'world', 'sports', 'india', 'business', 'features',
              'health', 'crime', 'lifestyle', 'environment', 'trending', 'technology', 
              'motor', 'spiritual', 'women', 'agriculture']
for category in categories:
    print(f"Fetching category: {category}")
    for offset in tqdm(range(0, 1500, 50),  desc="Epoch", unit="iter"):  # example: first 500 pages (50 items per page)
        batch = fetch_puthiyathalaimurai_api(category, offset=offset, limit=50)
        if not batch:
            break  # Stop if no more data
        all_data.extend(batch)

# Save to CSV
os.makedirs("dataset", exist_ok=True)
date_str = datetime.now().strftime("%Y-%m-%d")
df = pd.DataFrame(all_data)
df.to_csv(f"dataset/puthiyathalaimurai_api_{date_str}_second.csv", index=False, encoding="utf-8-sig")


print(f"Collected {len(df)} headlines via API")
print(df.head(5))