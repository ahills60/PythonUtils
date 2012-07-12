# Template builder
#
#  Author: Andrew Hills (a.hills@sheffield.ac.uk)
# Version: 0.9 (01/11/2011 10:44)
import sys, os

nargs = len(sys.argv)

if nargs == 0:
    print "\nUsage: python build.py filename.ext output.ext\n"
    print "Compatible languages: Matlab, LaTeX, Python"
    sys.exit(0)
elif nargs == 2:
    inputFile = sys.argv[-1]
    outputFile = sys.argv[-1].rsplit('.', 1)[-1] + '-build.' + sys.argv[-1].rsplit('.', 1)[1]
elif nargs == 3:
    inputFile = sys.argv[-2]
    outputFile = sys.argv[-1]
else:
    print "\nError: Unable to parse input."
    print "\nUsage: python build.py filename.ext output.ext\n"
    print "Compatible languages: Matlab, LaTeX, Python"
    sys.exit(-1)

if inputFile[-4:].lower() == '.tex':
    # LaTeX Parser
    commentChar = "%"
elif inputFile[-2:].lower() == '.m':
    # Matlab Parser
    commentChar = "%"
elif inputFile[-3:].lower() == '.py':
    # Python Parser
    commentChar = "#"
else:
    print "\nError: Unable to determine programming language by extension\n"
    sys.exit(-1)

if not os.path.exists(inputFile):
    print "\nError: Unable to find '" + inputFile + "'.\n"
    sys.exit(-1)

# Input file exists beyond this point

fin = open(inputFile, 'r')
fout = open(outputFile, 'w')

for line in fin:
    # Analyse line:
    justCommand = line.strip()
    if len(justCommand) == 0:
        fout.write('\n')
        continue
    if justCommand[0] == commentChar:
        # Look further afield...
        justCommand = justCommand.split(commentChar)[1].strip()
        if justCommand[:10] == "INPUTFILE:":
            # Extract file and check it exists
            if os.path.exists(justCommand[10:]):
                # Open said file and add to fout
                preamble = line.split(commentChar)[0]
                fsec = open(justCommand[10:], 'r')
                for secline in fsec:
                    fout.write(preamble + secline)
                fsec.close()
            else:
                # Can't find file:
                print "Warning: Unable to find file '" + justCommand[10:] + "'."
            continue
        else:
            fout.write(line)
    else:
        fout.write(line)
print "\nParsing complete. Closing files..."
fout.close()
fin.close()
print "Files closed.\n"