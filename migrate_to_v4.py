import os
import json
import shutil
import sys

def migrate_task(task_dir):
    pages_json = os.path.join(task_dir, 'pages.json')
    if not os.path.exists(pages_json):
        return
        
    try:
        with open(pages_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {pages_json}: {e}")
        return
        
    version = data.get('version', 1)
    if version >= 4:
        print(f"Skipping {task_dir}, already Version {version}")
        return
        
    print(f"Migrating {task_dir} from V{version} to V4...")
    
    # Backup
    shutil.copy2(pages_json, pages_json + '.bak')
    
    # translations_map[lang][block_id] = {text, edited}
    translations_map = {}
    
    for page in data.get('pages', []):
        if version <= 2:
            # V2 or earlier: block.translation (string)
            target_lang = data.get('target_lang', 'ENG')
            if target_lang not in translations_map: translations_map[target_lang] = {}
            for block in page.get('blocks', []):
                if 'translation' in block:
                    if block['translation']:
                        translations_map[target_lang][block['id']] = {
                            "text": block['translation'],
                            "edited": True
                        }
                    del block['translation']
        elif version == 3:
            # V3: page.translations[block_id][lang] = {text, edited}
            page_trans = page.get('translations', {})
            for bid, langs in page_trans.items():
                for lang, trans in langs.items():
                    if lang not in translations_map: translations_map[lang] = {}
                    translations_map[lang][bid] = trans
            if 'translations' in page:
                del page['translations']
                
    # Save translations
    trans_dir = os.path.join(task_dir, 'translations')
    os.makedirs(trans_dir, exist_ok=True)
    for lang, trans in translations_map.items():
        with open(os.path.join(trans_dir, f"{lang}.json"), 'w', encoding='utf-8') as f:
            json.dump(trans, f, ensure_ascii=False, indent=2)
            
    # Update pages.json
    data['version'] = 4
    with open(pages_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Successfully migrated {task_dir} to V4.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 migrate_to_v4.py <work_dir_or_task_dir>")
        sys.exit(1)
        
    target = sys.argv[1]
    if not os.path.isdir(target):
        print(f"Error: {target} is not a directory")
        sys.exit(1)
        
    if os.path.exists(os.path.join(target, 'pages.json')):
        migrate_task(target)
    else:
        # Assume it's a work_dir containing multiple tasks
        for entry in os.listdir(target):
            full = os.path.join(target, entry)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, 'pages.json')):
                migrate_task(full)

if __name__ == "__main__":
    main()
