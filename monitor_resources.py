#!/usr/bin/env python3
"""
Resource Monitor - Check RAM and disk usage on the cluster.

Usage:
    python monitor_resources.py
    python monitor_resources.py --watch          # Continuous monitoring every 2s
    python monitor_resources.py --watch --pid <PID>  # Monitor specific process
"""

import argparse
import os
import shutil
import sys
import time

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed. Run: pip install psutil")
    sys.exit(1)


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def get_system_memory():
    """Get system memory statistics."""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "total": mem.total,
        "available": mem.available,
        "used": mem.used,
        "percent": mem.percent,
        "swap_total": swap.total,
        "swap_used": swap.used,
        "swap_percent": swap.percent,
    }


def get_disk_usage(paths=None):
    """Get disk usage for specified paths or common ones."""
    if paths is None:
        paths = ["/", "/home", "/tmp"]
    
    results = {}
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
            results[path] = {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": (usage.used / usage.total) * 100,
            }
        except (FileNotFoundError, PermissionError):
            continue
    return results


def get_process_memory(pid=None):
    """Get memory usage for a specific process or current process."""
    if pid is None:
        pid = os.getpid()
    
    try:
        proc = psutil.Process(pid)
        mem = proc.memory_info()
        return {
            "pid": pid,
            "name": proc.name(),
            "rss": mem.rss,  # Resident Set Size (physical memory)
            "vms": mem.vms,  # Virtual Memory Size
            "percent": proc.memory_percent(),
        }
    except psutil.NoSuchProcess:
        return None


def get_top_memory_processes(n=10):
    """Get top N processes by memory usage."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'memory_info']):
        try:
            info = proc.info
            if info['memory_info'] is not None:
                processes.append({
                    "pid": info['pid'],
                    "name": info['name'],
                    "memory_percent": info['memory_percent'],
                    "rss": info['memory_info'].rss,
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return sorted(processes, key=lambda x: x['rss'], reverse=True)[:n]


def print_status(pid_to_watch=None, show_processes=True):
    """Print current resource status."""
    print("\n" + "=" * 70)
    print(f"RESOURCE MONITOR - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # System Memory
    mem = get_system_memory()
    print("\n📊 SYSTEM MEMORY:")
    print(f"   Total:     {format_bytes(mem['total'])}")
    print(f"   Used:      {format_bytes(mem['used'])} ({mem['percent']:.1f}%)")
    print(f"   Available: {format_bytes(mem['available'])}")
    if mem['swap_total'] > 0:
        print(f"   Swap:      {format_bytes(mem['swap_used'])} / {format_bytes(mem['swap_total'])} ({mem['swap_percent']:.1f}%)")
    
    # Memory bar
    bar_len = 40
    filled = int(bar_len * mem['percent'] / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    color = "\033[91m" if mem['percent'] > 80 else "\033[93m" if mem['percent'] > 60 else "\033[92m"
    print(f"   [{color}{bar}\033[0m] {mem['percent']:.1f}%")
    
    # Disk Usage
    print("\n💾 DISK USAGE:")
    disk = get_disk_usage()
    for path, usage in disk.items():
        print(f"   {path}: {format_bytes(usage['free'])} free / {format_bytes(usage['total'])} ({usage['percent']:.1f}% used)")
    
    # Process memory (if watching specific PID)
    if pid_to_watch:
        proc_mem = get_process_memory(pid_to_watch)
        if proc_mem:
            print(f"\n🔍 WATCHED PROCESS (PID {pid_to_watch}):")
            print(f"   Name:    {proc_mem['name']}")
            print(f"   RSS:     {format_bytes(proc_mem['rss'])} (physical)")
            print(f"   VMS:     {format_bytes(proc_mem['vms'])} (virtual)")
            print(f"   % Mem:   {proc_mem['percent']:.2f}%")
        else:
            print(f"\n⚠️  Process {pid_to_watch} not found")
    
    # Top processes
    if show_processes:
        print("\n🔝 TOP 5 PROCESSES BY MEMORY:")
        top_procs = get_top_memory_processes(5)
        print(f"   {'PID':<8} {'Name':<25} {'RSS':<12} {'% Mem':<8}")
        print("   " + "-" * 55)
        for proc in top_procs:
            print(f"   {proc['pid']:<8} {proc['name'][:24]:<25} {format_bytes(proc['rss']):<12} {proc['memory_percent']:.2f}%")
    
    # Warning if memory is critical
    if mem['percent'] > 90:
        print("\n⚠️  \033[91mCRITICAL: Memory usage above 90%! OOM kill likely!\033[0m")
    elif mem['percent'] > 80:
        print("\n⚠️  \033[93mWARNING: Memory usage above 80%\033[0m")


def main():
    parser = argparse.ArgumentParser(description="Monitor system resources")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", "-i", type=float, default=2.0, help="Monitoring interval in seconds")
    parser.add_argument("--pid", "-p", type=int, help="PID of process to watch")
    parser.add_argument("--no-processes", action="store_true", help="Don't show top processes")
    args = parser.parse_args()
    
    if args.watch:
        print("Starting continuous monitoring (Ctrl+C to stop)...")
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                print_status(pid_to_watch=args.pid, show_processes=not args.no_processes)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        print_status(pid_to_watch=args.pid, show_processes=not args.no_processes)


if __name__ == "__main__":
    main()
