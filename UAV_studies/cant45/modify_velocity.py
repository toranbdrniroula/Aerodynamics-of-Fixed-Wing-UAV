import sys

def modify_velocity(file_path, Vx, Vy):
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Loop through the lines and find the line containing 'internalField uniform'
    for i, line in enumerate(content):
        if "internalField   uniform (20 0 0);  " in line:
            # Modify the line with the new Vx and Vy values
            content[i] = f"internalField   uniform ({Vx} {Vy} 0);\n"
            break  # Exit after modifying the first match

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(content)
        
def modify_force(file_path, Vx, Vy):
    with open(file_path, 'r') as file:
        content = file.readlines()
    Vx = float(Vx)
    Vy = float(Vy)
    y = Vx/(Vx**2+Vy**2)**0.5
    x = Vy/(Vx**2+Vy**2)**0.5
    # Loop through the lines and find the line containing 'internalField uniform'
    for i, line in enumerate(content):
        if "    liftDir         (0 1 0);" in line:
            # Modify the line with the new Vx and Vy values
            content[i] = f"    liftDir         ({-1*x} {y} 0);\n"
            break  # Exit after modifying the first match

    for i, line in enumerate(content):
        if "    dragDir         (1 0 0);" in line:
            # Modify the line with the new Vx and Vy values
            content[i] = f"    dragDir         ({y} {x} 0);\n"
            break  # Exit after modifying the first match
            
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(content)

if __name__ == "__main__":
    # Get the file path and velocity components from command line arguments
    case_dir = sys.argv[1]
    Vx = sys.argv[2]
    Vy = sys.argv[3]
    
    # Path to the 0/U file
    file_path = f"{case_dir}/0/U"
    file_path1 = f"{case_dir}/system/forceCoeffs"    
    # Modify the velocity
    modify_velocity(file_path, Vx, Vy)
    modify_force(file_path1, Vx, Vy)
    print(f"Velocity in {file_path} updated to ({Vx}, {Vy}, 0)")
