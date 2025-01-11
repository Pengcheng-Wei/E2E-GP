import sys
import warnings   
warnings.filterwarnings('ignore') 
from modelUtils import partitioning
from collections import OrderedDict
P = OrderedDict([
    ('graph_name', ''),
    ('es_patience', 100),
    ('patience', 50),
    ('device', "cuda:0"),
    ('n_channels', 32),
    ('alpha', 1),
    ('learning_rate', 5e-3),

])


if __name__ == '__main__':
    n_clust = int(sys.argv[2])
    P['graph_name'] = sys.argv[1]
    P['alpha'] = float(sys.argv[3])
    P['device'] = sys.argv[4]
    print("We will partition '" + P['graph_name'] + "' into {} partitions ....".format(str(n_clust)), flush=True)
    result = partitioning(P, n_clust)


