import os
import re
import shutil

# Configuration
TARGET_EXTENSIONS = ('.c', '.cpp', '.h', '.hpp')
PLACEHOLDER = 'std::placeholders::_1'
INCLUDE_STATEMENT = '#include <functional>'
BACKUP_SUFFIX = '.bak'
LOG_FILE = 'bind2nd_replacement.log'

def backup_file(file_path):
    backup_path = file_path + BACKUP_SUFFIX
    #shutil.copyfile(file_path, backup_path)
    print(f"Backup created: {backup_path}")

def add_include(content):
    include_pattern = re.compile(r'#include\s*<functional>')
    if include_pattern.search(content):
        return content  # Include already present

    # Find the last include statement to insert after
    includes = list(re.finditer(r'#include\s*[<"].*[>"]', content))
    if includes:
        last_include = includes[-1]
        insert_pos = last_include.end()
        new_content = content[:insert_pos] + '\n' + INCLUDE_STATEMENT + content[insert_pos:]
        return new_content
    else:
        # If no includes are present, add at the top
        return INCLUDE_STATEMENT + '\n' + content

def replace_bind2nd(content):
    """
    Replaces bind2nd with std::bind and inserts std::placeholders::_1.

    Example:
    std::bind2nd(std::greater<int>(), 3)
    becomes
    std::bind(std::greater<int>(), std::placeholders::_1, 3)
    """

    # Regex to find bind2nd usage
    # This pattern looks for bind2nd(function, value)
    pattern = re.compile(r'bind2nd\s*\(\s*(.*)\s*,\s*(.*)\s*\)')

    def replacer(match):
        func = match.group(1).strip()
        value = match.group(2).strip()
        # Replace with std::bind(func, std::placeholders::_1, value)
        return f'bind({func}, {PLACEHOLDER}, {value})'

    new_content, num_subs = pattern.subn(replacer, content)
    return new_content, num_subs

def process_file(file_path, log):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    original_content = content

    # Replace bind2nd with std::bind
    content, num_replacements = replace_bind2nd(content)

    if num_replacements > 0:
        # Add #include <functional> if not present
        content = add_include(content)

        # Backup the original file
        backup_file(file_path)

        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

        # Log the changes
        log.write(f"{file_path}: {num_replacements} replacements made.\n")
        print(f"Modified: {file_path} | Replacements: {num_replacements}")
    else:
        print(f"No replacements needed: {file_path}")

def traverse_and_process(root_dir):
    with open(LOG_FILE, 'w', encoding='utf-8') as log:
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(TARGET_EXTENSIONS):
                    file_path = os.path.join(subdir, file)
                    try:
                        process_file(file_path, log)
                    except Exception as e:
                        log.write(f"Error processing {file_path}: {e}\n")
                        print(f"Error processing {file_path}: {e}")

    print(f"\nProcessing complete. See '{LOG_FILE}' for details.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replace bind2nd with std::bind in C/C++ files.")
    parser.add_argument('directory', help="Path to the target directory.")
    args = parser.parse_args()

    target_directory = args.directory

    if not os.path.isdir(target_directory):
        print(f"Error: The directory '{target_directory}' does not exist.")
        exit(1)

    traverse_and_process(target_directory)

