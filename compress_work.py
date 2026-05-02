import os
import shutil
import argparse
from PIL import Image
import cv2
import numpy as np

from manga_translator_lite.pipeline.schema import load_workspace, save_workspace
from manga_translator_lite.utils import cv2_imread, cv2_imwrite

def find_original_image(in_dir: str, task_name: str, original_filename: str) -> str:
    """Find the original image path robustly."""
    path1 = os.path.join(in_dir, task_name, original_filename)
    if os.path.isfile(path1):
        return path1
    path2 = os.path.join(in_dir, original_filename)
    if os.path.isfile(path2):
        return path2
    return ""

def process_workspace(in_dir: str, work_dir: str, task_name: str):
    ws_root = os.path.join(work_dir, task_name)
    try:
        ws = load_workspace(ws_root)
    except Exception as e:
        print(f"Skipping {task_name}: could not load workspace ({e})")
        return

    changed = False
    for page in ws.pages:
        orig_path = find_original_image(in_dir, task_name, page.original)
        if not orig_path:
            print(f"  [Warning] Original image not found for {page.original}")
            continue
            
        clean_abs = os.path.join(ws_root, page.clean)
        if not os.path.isfile(clean_abs):
            print(f"  [Warning] Clean image not found: {clean_abs}")
            continue
            
        _, orig_ext = os.path.splitext(orig_path)
        orig_ext = orig_ext.lower()
        if not orig_ext:
            orig_ext = '.png'
            
        new_clean_name = f"{page.index:04d}_{os.path.splitext(page.original)[0]}{orig_ext}"
        new_clean_rel = os.path.join("clean", new_clean_name)
        new_clean_abs = os.path.join(ws_root, new_clean_rel)
        
        orig_size = os.path.getsize(orig_path)
        clean_size = os.path.getsize(clean_abs)
        
        # 1. No text / No blocks -> should be identical to original
        if getattr(page, 'no_text', False) or not page.blocks:
            if clean_abs != new_clean_abs or clean_size != orig_size:
                print(f"  -> {page.original}: No text detected. Replacing with exact copy of original.")
                shutil.copy2(orig_path, new_clean_abs)
                if clean_abs != new_clean_abs and os.path.exists(clean_abs):
                    os.remove(clean_abs)
                page.clean = new_clean_rel
                changed = True
            else:
                print(f"  -> {page.original}: Already a perfect copy.")
            continue
            
        # 2. Inpainted images -> should have reasonable size and correct format
        # If the file extension is already correct AND the size is close to original, skip.
        if clean_abs == new_clean_abs and clean_size <= orig_size * 1.5:
            print(f"  -> {page.original}: Size already reasonable ({clean_size // 1024}KB vs orig {orig_size // 1024}KB). Skipping.")
            continue
            
        try:
            pil_orig = Image.open(orig_path)
        except Exception as e:
            print(f"  [Warning] Failed to open original image {orig_path}: {e}")
            continue
            
        img_bgr = cv2_imread(clean_abs)
        if img_bgr is None:
            print(f"  [Warning] Failed to read clean image {clean_abs}")
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        print(f"  -> {page.original}: Recompressing (current: {clean_size // 1024}KB, orig: {orig_size // 1024}KB)...")
        if pil_orig.format == 'JPEG':
            pil_img = Image.fromarray(img_rgb)
            pil_img.format = pil_orig.format
            pil_img.info = pil_orig.info
            try:
                pil_img.save(new_clean_abs, format=pil_orig.format, quality='keep', subsampling='keep')
            except Exception as e:
                print(f"     Fallback to cv2 for {page.original}: {e}")
                cv2_imwrite(new_clean_abs, img_bgr)
        else:
            cv2_imwrite(new_clean_abs, img_bgr)
            
        # Clean up old file if name changed
        if clean_abs != new_clean_abs and os.path.exists(clean_abs):
            os.remove(clean_abs)
            
        page.clean = new_clean_rel
        changed = True
            
    if changed:
        save_workspace(ws)
        print(f"Updated pages.json for {task_name}\n")
    else:
        print(f"No changes needed for {task_name}\n")

def main():
    parser = argparse.ArgumentParser(description="Compress existing work files based on original images.")
    parser.add_argument("--in-dir", "-i", type=str, default="in", help="Input directory (default: 'in')")
    parser.add_argument("--work-dir", "-w", type=str, default="work", help="Work directory (default: 'work')")
    args = parser.parse_args()
    
    in_dir = os.path.abspath(args.in_dir)
    work_dir = os.path.abspath(args.work_dir)
    
    if not os.path.isdir(work_dir):
        print(f"Work directory not found: {work_dir}")
        return
        
    for task_name in os.listdir(work_dir):
        ws_root = os.path.join(work_dir, task_name)
        if os.path.isdir(ws_root) and os.path.isfile(os.path.join(ws_root, "pages.json")):
            print(f"Processing task: {task_name}")
            process_workspace(in_dir, work_dir, task_name)

if __name__ == "__main__":
    main()
