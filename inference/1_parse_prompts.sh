#!/bin/bash

# Check if the number of arguments is correct
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input-file> <number-of-parts>"
    exit 1
fi

# Input file and number of parts
input_file=$1
num_parts=$2

# Get total number of lines (words) in the input file
total_lines=$(wc -l < "$input_file")

# Calculate the approximate number of lines (words) per part
lines_per_part=$((total_lines / num_parts))
echo "Total lines: $total_lines"
echo "Lines per part: $lines_per_part"
echo "Remaining lines: $((total_lines % num_parts))"
if [ $((total_lines % num_parts)) -ne 0 ]; then
    echo "Error: The number of lines is not divisible by the number of parts"
    exit 1
fi

# Split the file based on the calculated number of lines per part
split -l "$lines_per_part" "$input_file" part_

# Rename the output files to have a .txt extension and ensure correct numbering
a=0
for i in part_*; do
    mv "$i" "prompts/prompt_$a.txt"
    a=$((a + 1))
done
echo "Done"
