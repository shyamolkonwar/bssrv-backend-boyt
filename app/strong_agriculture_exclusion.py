"""
Script to add stronger exclusion language about agriculture programs to the knowledge base
"""

import os

def add_strong_exclusion_statements():
    # Get the path to the knowledge base file
    file_path = os.path.join(os.path.dirname(__file__), 'bssrv_general_info.txt')
    
    # Read the current content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Add even stronger exclusion statements at the beginning of the file
    strong_disclaimer = """# BSSRV University General Information

## IMPORTANT: Available Programs
BSSRV University ONLY offers B.Tech programs in engineering disciplines. 
THE UNIVERSITY DOES NOT OFFER:
- BSc Agriculture
- ANY agriculture-related programs
- Agriculture admissions
- Farming or agricultural science degrees

There is NO website, email, or application form for agricultural programs at BSSRV.
Any information about agricultural admissions at BSSRV is incorrect.
"""
    
    # Replace the current header with our stronger disclaimer
    updated_content = content.replace("# BSSRV University General Information", strong_disclaimer, 1)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(updated_content)
    
    print("Knowledge base updated with stronger exclusion statements about agriculture programs!")

if __name__ == "__main__":
    add_strong_exclusion_statements() 