"""
Script to update the bssrv_general_info.txt file to remove Agriculture from department listings
and add explicit note that BSc Agriculture is not currently offered.
"""

import os
import re

def update_knowledge_base():
    # Get the path to the knowledge base file
    file_path = os.path.join(os.path.dirname(__file__), 'bssrv_general_info.txt')
    
    # Read the current content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Remove Agriculture from the departments list
    departments_pattern = r'(## Departments.*?The university houses several departments catering to diverse academic interests:\s*Arts\s*Engineering and Technology\s*Sciences\s*Management Science\s*Medicine\s*)Agriculture\s*(Fishery Science)'
    updated_content = re.sub(departments_pattern, r'\1\2', content, flags=re.DOTALL)
    
    # Add a note about programs offered
    programs_pattern = r'(# BSSRV University General Information.*?## Overview)'
    programs_note = r'\1\n\n## Programs Offered\nBSSRV University currently offers B.Tech programs in various engineering disciplines. The university does not currently offer BSc Agriculture or any agriculture-related undergraduate programs.\n'
    updated_content = re.sub(programs_pattern, programs_note, updated_content, flags=re.DOTALL)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(updated_content)
    
    print("Knowledge base updated successfully!")
    print("Agriculture removed from departments list.")
    print("Added explicit note about BSc Agriculture not being offered.")

if __name__ == "__main__":
    update_knowledge_base() 