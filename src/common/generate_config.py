import argparse

parser = argparse.ArgumentParser(description='Get dataset')
parser.add_argument('--dataset', type=str, required=True, help='dataset name')
parser.add_argument('--time', type=str, required=True, help='current time')
args = parser.parse_args()
print(f"Generating config for dataset: {args.dataset} at time {args.time}")

# Read template
with open("config.template", "r") as f:
    template = f.read()

# Get dataset
variables = {
    "dataset": args.dataset,
    "time": args.time
}

# Replace placeholders
output_content = template.format(**variables)  # Unpacks the dictionary

# Write output
with open("config.yaml", "w") as f:
    f.write(output_content)
