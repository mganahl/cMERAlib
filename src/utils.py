import re
import copy

def get_checkpoint_number(string):
    """
    takes as input string a cMERA checkpoint filename and returns
    its iteration number
    """
    n=len(string)-1
    while(string[n]!='.'):
        n-=1
    n1=copy.copy(n)
    while(string[n]!='_'):
        n-=1
    return int(string[n+1:n1])

def convert(val):

    """
    converts an input str() val into its numeric type
    see the conversion rules below
    
    """
    if val=='True':
        return True
    elif val=='False':
        return False
    elif val=='None':
        return None
    else:
        types=[int,float,str]
    for t in types:
        try:
            return t(val)
        except ValueError:
            pass
        
def read_parameters(filename):
    """
    read parameters from a file "filename"
    the file format is assumed to be 

    parameter1 value1
    parameter2 value2
        .
        .
        .

    or 

    parameter1: value1
    parameter2: value2

    Returns:
    python dict() containing mapping parameter-name to its value
    """
    
    params={}
    with open(filename, 'r') as f:
        for line in f:
            if '[' not in line:
                params[line.replace(':','').split()[0]]=convert(line.replace(':','').split()[1])
            else:
                s=re.sub('[\[\]:,\']','',line).split()
                params[s[0]]=[convert(t) for t in s[1::]]
    return params

