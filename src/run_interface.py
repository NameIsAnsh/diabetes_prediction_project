#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launcher script for the Diabetes Prediction Interface
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """
    Launch the Streamlit interface for the Diabetes Prediction System
    """
    # Get the current directory
    current_dir = Path.cwd()
    
    # Path to the interface script
    interface_script = current_dir / "src" / "create_interface.py"
    
    # Check if the script exists
    if not interface_script.exists():
        print(f"Error: Interface script not found at {interface_script}")
        sys.exit(1)
    
    # Launch Streamlit
    print("Starting Diabetes Prediction Interface...")
    print(f"Access the interface at http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    # Run Streamlit
    subprocess.run(["streamlit", "run", str(interface_script)])

if __name__ == "__main__":
    main()
