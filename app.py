import sys
from .module.cumulative_images import cumulative_images
from .module.find_defect import find_defect

def run():
    modules = {
        "CI": cumulative_images,
        "FD": find_defect,
    }
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("the dice analysis need the name of the module to run")
        print("to run module xxx: python -m dice_analysis xxx")
        all_modules = ", ".join(list(modules.keys()))
        print(f"List of modules available: {all_modules}")
        sys.exit()

    
    if sys.argv[1] in modules.keys():
        modules[sys.argv[1]](sys.argv[2:])
    else:
        print(f"Module {sys.argv[1]} doesn't exist")
        all_modules = ", ".join(list(modules.keys()))
        print(f"List of modules available: {all_modules}")
        sys.exit()


            



