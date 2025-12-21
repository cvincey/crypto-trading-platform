#!/usr/bin/env python3
"""
Multi-Symbol Paper Trading Launcher.

Starts paper trading for ALL enabled strategies in parallel.
Each strategy runs as a separate background process.

Usage:
    python scripts/run_multi_paper_trading.py           # Start all enabled
    python scripts/run_multi_paper_trading.py --status  # Check status
    python scripts/run_multi_paper_trading.py --stop    # Stop all
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

LOG_DIR = Path("notes/test_logs")
PID_FILE = LOG_DIR / "paper_trading_pids.txt"


def load_config() -> dict:
    """Load paper trading configuration."""
    config_path = Path("config/paper_trading.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)["paper_trading"]


def get_enabled_strategies(config: dict) -> list[tuple[str, dict]]:
    """Get list of enabled strategies."""
    strategies = config.get("strategies", {})
    enabled = []
    for name, strat_config in strategies.items():
        if strat_config.get("enabled", False):
            enabled.append((name, strat_config))
    return enabled


def start_all():
    """Start paper trading for all enabled strategies."""
    config = load_config()
    enabled = get_enabled_strategies(config)
    
    if not enabled:
        console.print("[red]No enabled strategies found[/red]")
        return
    
    console.print(Panel(
        f"[bold blue]Starting Multi-Symbol Paper Trading[/bold blue]\n\n"
        f"Enabled strategies: {len(enabled)}",
        expand=False,
    ))
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pids = []
    
    table = Table(title="Starting Strategies")
    table.add_column("Strategy", style="cyan")
    table.add_column("Symbol")
    table.add_column("Status")
    table.add_column("Log File")
    
    for name, strat_config in enabled:
        symbol = strat_config.get("symbol", "???")
        log_file = LOG_DIR / f"paper_{name}.log"
        
        # Start process
        cmd = [
            sys.executable,
            "scripts/run_paper_trading.py",
            "--strategy", name,
        ]
        
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path.cwd(),
            )
        
        pids.append((name, proc.pid))
        table.add_row(name, symbol, f"✓ PID {proc.pid}", str(log_file))
    
    console.print(table)
    
    # Save PIDs
    with open(PID_FILE, "w") as f:
        for name, pid in pids:
            f.write(f"{name},{pid}\n")
    
    console.print(f"\n[green]Started {len(pids)} paper trading processes[/green]")
    console.print(f"[dim]PIDs saved to {PID_FILE}[/dim]")
    console.print("\n[bold]Monitor with:[/bold]")
    console.print("  tail -f notes/test_logs/paper_*.log")
    console.print("\n[bold]Stop with:[/bold]")
    console.print("  python scripts/run_multi_paper_trading.py --stop")


def show_status():
    """Show status of running paper trading processes."""
    if not PID_FILE.exists():
        console.print("[yellow]No paper trading session found[/yellow]")
        return
    
    table = Table(title="Paper Trading Status")
    table.add_column("Strategy", style="cyan")
    table.add_column("PID")
    table.add_column("Status")
    table.add_column("Last Log Line")
    
    with open(PID_FILE) as f:
        for line in f:
            name, pid = line.strip().split(",")
            pid = int(pid)
            
            # Check if process is running
            try:
                os.kill(pid, 0)
                status = "[green]Running[/green]"
            except OSError:
                status = "[red]Stopped[/red]"
            
            # Get last log line
            log_file = LOG_DIR / f"paper_{name}.log"
            last_line = ""
            if log_file.exists():
                with open(log_file) as lf:
                    lines = lf.readlines()
                    if lines:
                        last_line = lines[-1].strip()[:50] + "..."
            
            table.add_row(name, str(pid), status, last_line)
    
    console.print(table)


def stop_all():
    """Stop all paper trading processes."""
    if not PID_FILE.exists():
        console.print("[yellow]No paper trading session found[/yellow]")
        return
    
    stopped = 0
    with open(PID_FILE) as f:
        for line in f:
            name, pid = line.strip().split(",")
            pid = int(pid)
            
            try:
                os.kill(pid, signal.SIGTERM)
                console.print(f"[yellow]Stopped {name} (PID {pid})[/yellow]")
                stopped += 1
            except OSError:
                console.print(f"[dim]{name} already stopped[/dim]")
    
    PID_FILE.unlink()
    console.print(f"\n[green]Stopped {stopped} processes[/green]")


def main():
    parser = argparse.ArgumentParser(description="Multi-symbol paper trading")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--stop", action="store_true", help="Stop all")
    args = parser.parse_args()
    
    if args.status:
        show_status()
    elif args.stop:
        stop_all()
    else:
        start_all()


if __name__ == "__main__":
    main()
