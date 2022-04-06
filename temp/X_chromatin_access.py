import pandas as pd
import pyranges as pr
import pyfaidx

class ChromatinAceess:
    def __init__(self, domain):
        self.target = domain
    
    def run_func(self, data, file_prefix):
        
        db = self.db_load()


        