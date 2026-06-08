import sys
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table

from ..config import Config, LLMProvider
from ..translators import build_translator
from ..pipeline.schema import Workspace, Translation, load_translations, save_translations

console = Console()

async def run_proofread_check(
    workspace: Workspace,
    cfg: Config,
    yes: bool = False,
) -> bool:
    """Run proofreading checks on workspace translations.
    
    Returns True if rendering should proceed, False if aborted.
    """
    if cfg.translator.provider == LLMProvider.none:
        console.print("[yellow]Translator provider is 'none'; skipping proofreading check.[/yellow]")
        return True

    # 1. Load translations
    translations = load_translations(workspace.root, workspace.target_lang)
    
    # 2. Gather translation items
    # We want a list of (block_id, original_text, current_translation)
    items: List[Tuple[str, str, str]] = []
    # Build a lookup for original texts
    block_lookup = {}
    for page in workspace.pages:
        if page.no_text:
            continue
        for block in page.blocks:
            block_lookup[block.id] = block.text
            t = translations.get(block.id)
            if t and t.text and t.text.strip():
                items.append((block.id, block.text, t.text))

    if not items:
        console.print(f"[yellow]No translated blocks found in workspace '{workspace.task_name}' for proofreading.[/yellow]")
        return True

    console.print(f"[cyan]Proofreading {len(items)} translated block(s) using LLM ({cfg.translator.model})...[/cyan]")
    
    # Instantiate translator and proofread
    translator = build_translator(cfg.translator)
    if not hasattr(translator, "proofread"):
        console.print("[yellow]Selected translator does not support proofreading. Skipping.[/yellow]")
        return True

    try:
        raw_suggestions = await translator.proofread(items)
    except Exception as e:
        console.print(f"[red]Proofreading check failed: {e}. Proceeding without check.[/red]")
        return True

    # Filter out suggestions that are identical or only differ by punctuation/whitespace/symbols
    suggestions = []
    for sug in raw_suggestions:
        bid = sug["id"]
        curr = translations.get(bid).text if translations.get(bid) else ""
        if sug["suggestion"].strip() == curr.strip():
            continue
        # Compare alphanumeric characters to ignore punctuation-only differences
        clean_sug = "".join(c for c in sug["suggestion"] if c.isalnum())
        clean_curr = "".join(c for c in curr if c.isalnum())
        if clean_sug == clean_curr:
            continue
        suggestions.append(sug)

    if not suggestions:
        console.print("[green]No spelling or fluency issues found! Everything looks great.[/green]")
        return True

    console.print(f"[yellow]Found {len(suggestions)} proofreading suggestion(s):[/yellow]\n")

    # Present suggestions in a detailed list format
    for idx, sug in enumerate(suggestions, 1):
        bid = sug["id"]
        orig = block_lookup.get(bid, "")
        curr = translations.get(bid).text if translations.get(bid) else ""
        
        console.print(f"[bold magenta]Suggestion #{idx} (Block ID: {bid})[/bold magenta]")
        console.print(f"  [bold cyan]Original Text:[/bold cyan]\n    {orig.replace(chr(10), chr(10) + '    ')}")
        console.print(f"  [bold yellow]Current Translation:[/bold yellow]\n    {curr.replace(chr(10), chr(10) + '    ')}")
        console.print(f"  [bold green]Polished Suggestion:[/bold green]\n    {sug['suggestion'].replace(chr(10), chr(10) + '    ')}")
        console.print(f"  [bold blue]Reason:[/bold blue] {sug.get('reason', 'N/A')}")
        console.print("-" * 50)
    print()

    # 3. Handle user response
    apply_all = False
    interactive = False
    
    if yes:
        apply_all = True
    elif not sys.stdin.isatty():
        console.print("[yellow]Non-interactive terminal, but suggestions found. Proceeding with original translations without auto-applying changes.[/yellow]")
        return True
    else:
        # Prompt user
        while True:
            try:
                choice = input("Do you want to apply these suggestions? [y]es to all / [n]o to all / [i]nteractive review / [q]uit: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[red]Rendering aborted by user.[/red]")
                return False
                
            if choice in ("y", "yes"):
                apply_all = True
                break
            elif choice in ("n", "no", ""):
                # Default is no, do not apply but proceed with rendering
                break
            elif choice in ("i", "interactive"):
                interactive = True
                break
            elif choice in ("q", "quit", "c", "cancel"):
                console.print("[red]Rendering aborted by user.[/red]")
                return False

    if apply_all:
        # Apply all suggestions
        updated_count = 0
        for sug in suggestions:
            bid = sug["id"]
            if bid in translations:
                translations[bid].text = sug["suggestion"]
                translations[bid].edited = True
                updated_count += 1
        if updated_count > 0:
            save_translations(workspace.root, workspace.target_lang, translations)
            console.print(f"[green]Applied {updated_count} proofreading suggestion(s) and saved translations.[/green]")
    elif interactive:
        updated_count = 0
        for sug in suggestions:
            bid = sug["id"]
            orig = block_lookup.get(bid, "")
            curr = translations.get(bid).text if translations.get(bid) else ""
            
            console.print("-" * 60)
            console.print(f"[bold cyan]Block ID:[/bold cyan] {bid}")
            console.print(f"[bold cyan]Original Text:[/bold cyan] {orig}")
            console.print(f"[bold yellow]Current Translation:[/bold yellow] {curr}")
            console.print(f"[bold green]Suggestion:[/bold green] {sug['suggestion']}")
            console.print(f"[bold blue]Reason:[/bold blue] {sug.get('reason', 'N/A')}")
            
            act = ""
            while True:
                try:
                    act = input("Action: [y]es / [n]o / [e]dit / [q]uit review (keep remaining as-is) [default: y]: ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    act = "q"
                    console.print("\n[yellow]Interactive review stopped.[/yellow]")
                    break
                    
                if act in ("y", "yes", ""):
                    translations[bid].text = sug["suggestion"]
                    translations[bid].edited = True
                    updated_count += 1
                    console.print("[green]Applied suggestion.[/green]")
                    break
                elif act in ("n", "no"):
                    console.print("[yellow]Kept current translation.[/yellow]")
                    break
                elif act in ("e", "edit"):
                    try:
                        new_text = input("Enter new translation: ").strip()
                    except (KeyboardInterrupt, EOFError):
                        new_text = ""
                    if new_text:
                        translations[bid].text = new_text
                        translations[bid].edited = True
                        updated_count += 1
                        console.print(f"[green]Saved custom translation: {new_text}[/green]")
                    else:
                        console.print("[yellow]Custom translation empty. Kept current.[/yellow]")
                    break
                elif act in ("q", "quit"):
                    console.print("[yellow]Interactive review stopped.[/yellow]")
                    break
            
            if act in ("q", "quit"):
                break
        
        if updated_count > 0:
            save_translations(workspace.root, workspace.target_lang, translations)
            console.print(f"[green]Applied {updated_count} translation(s) and saved.[/green]")

    return True
