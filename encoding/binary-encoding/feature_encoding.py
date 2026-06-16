import argparse
import json
import logging
import sys 
import time 
from datetime import datetime
from pathlib import Path 

STAGE = "encoding"

EXPECTED_PROTOCOLS = [0,6,17]

TECHNIQUE = "onehot"

IDENTIFIED_COLS = ("Dst_Port")