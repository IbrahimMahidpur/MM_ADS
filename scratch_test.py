import os
import sys
import logging

# Set up paths and environment
sys.path.insert(0, os.path.abspath('src'))
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('debug_output.log', encoding='utf-8')
fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

from multimodal_ds.core.llm_client import chat

response = chat([{"role": "user", "content": "Hello"}], model="opencode/minimax-m2.5-free")
print(response)
