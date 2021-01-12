import argparse
import os
import datetime



























# Initialize parser

parser = argparse.ArgumentParser()

parser.add_argument("inputfolder", help = "Input File")
parser.add_argument("outputfolder", help = "Output File")

args = parser.parse_args()


tc_name = args.test_case_file_path
o_name = args.output_file_path

main(f_name, tc_name, o_name)
