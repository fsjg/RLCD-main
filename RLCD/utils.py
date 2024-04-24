import argparse
from build_graph import build_graph

class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        # 3162,102,1709
        self.add_argument('--exer_n', type=int, default = 835,
                          help='The number for exercise.')
        self.add_argument('--knowledge_n', type=int, default = 835,
                          help='The number for knowledge concept.')
        self.add_argument('--student_n', type=int, default = 10000,
                          help='The number for student.')
        self.add_argument('--gpu', type=int, default = 0,
                          help='The id of gpu, e.g. 0.')
        self.add_argument('--epoch_n', type=int, default = 30,
                          help='The epoch number of training')
        self.add_argument('--lr', type=float, default = 0.002,
                          help='Learning rate')
        self.add_argument('--test', action='store_true',
                          help='Evaluate the model on the testing set in the training process.')

def construct_local_map(args):
    local_map = {
        'directed_g': build_graph('direct', args.knowledge_n),
        'undirected_g': build_graph('undirect', args.knowledge_n),
        # 'similar_exer_g': build_graph('similar_exer',args.exer_n),
        'similar_exer_g': build_graph('similar_exer',args.exer_n)[0],
        'similar_stu_g': build_graph('similar_stu',args.student_n),
    }
    return local_map, build_graph('similar_exer',args.exer_n)[1] # 计算习题间余弦相似度
    # return local_map