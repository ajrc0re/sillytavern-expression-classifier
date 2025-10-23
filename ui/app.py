"""Streamlit UI wrapper for the batch classifier."""
import subprocess
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "data" / "sorted"

st.set_page_config(page_title="Anime Emotion Sorter")
st.title("Anime Emotion Sorter")

input_dir = st.text_input("Input folder", str(DEFAULT_INPUT))
output_dir = st.text_input("Output folder", str(DEFAULT_OUTPUT))

if st.button("Run Classification"):
    st.write("Running batch classifier...")
    result = subprocess.run(
        [
            "python",
            str(ROOT / "scripts" / "classify_batch.py"),
            "--input-dir",
            input_dir,
            "--output-dir",
            output_dir,
        ],
        cwd=str(ROOT / "scripts"),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        st.success("Classification finished.")
    else:
        st.error("Classification failed. Check logs for details.")
    st.text(result.stdout)
    st.text(result.stderr)
