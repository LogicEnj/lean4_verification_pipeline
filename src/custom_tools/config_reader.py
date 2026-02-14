import inspect
import os
import yaml
from collections.abc import Iterable

def is_iterable(obj):
    return isinstance(obj, Iterable)

def get_caller_filename():
    # Traverse up the call stack to find the first frame outside this module
    current_frame = inspect.currentframe()
    try:
        while current_frame:
            caller_frame = current_frame.f_back
            if caller_frame is None:
                break  # No more frames to traverse
            # Get the filename of the caller's module
            caller_filename = caller_frame.f_globals.get('__file__', 'Unknown')
            # Check if the caller is outside this module
            caller_module = caller_frame.f_globals.get('__name__')
            if caller_module != __name__:  # Ensure it's not this module
                return os.path.basename(caller_filename)
            current_frame = caller_frame
        return None
    finally:
        # Ensure we clean up the frame reference to avoid memory leaks
        del current_frame

def equals_or_is_contained(element, element_or_list) :
    return element == element_or_list or is_iterable(element_or_list) and element in element_or_list

def get_input_file_or_folder(config_file, index = 0) :
    with open(config_file) as f:
        config = yaml.safe_load(f)

    calling_module_file_name = get_caller_filename()

    input_file_list = [
        file_info['path'] for file_info in config["files"]
        if equals_or_is_contained(calling_module_file_name, file_info["consumed_by"]) 
    ]
    
    return input_file_list[index]

def get_output_file_or_folder(config_file, index = 0) :
    with open(config_file) as f:
        config = yaml.safe_load(f)

    calling_module_file_name = get_caller_filename()

    output_file_list = [
        file_info['path'] for file_info in config["files"]
        if equals_or_is_contained(calling_module_file_name, file_info["produced_by"])
    ]
  
    return output_file_list[index]
