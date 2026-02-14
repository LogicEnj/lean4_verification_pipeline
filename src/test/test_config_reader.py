from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder 

input_file = get_input_file_or_folder('config.yaml')

print('Getting input file name from config.yaml:', input_file)

output_file = get_output_file_or_folder('config.yaml')

print('Getting output file name from config.yaml:', output_file)
