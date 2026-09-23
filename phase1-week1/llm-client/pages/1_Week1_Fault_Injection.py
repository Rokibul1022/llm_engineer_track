"""Week 1 Module: Fault-Injection & Resilient Retry Policy Demo.

Reuses the fault-injection scenarios and rendering logic from streamlit_demo.py.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import streamlit as st
from streamlit_demo import render_fault_injection_ui

render_fault_injection_ui(set_config=False)
