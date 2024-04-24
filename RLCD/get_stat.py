import torch
from model_ncd import Net
from utils import CommonArgParser, construct_local_map
import csv


student_n = 10000
def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()

def get_status(args,local_map):
    '''
    An example of getting student's knowledge status
    :return:
    '''
    net = Net(args, local_map)
    load_snapshot(net, 'RLCD-main/model/model_epoch30')       # load model
    net.eval()
    with open('RLCD-main/result/stu_stat_junyi.csv', 'w', encoding='utf8') as output_file:
        for stu_id in range(student_n):
            # get knowledge status of student with stu_id (index)
            status = net.get_knowledge_status(torch.LongTensor([stu_id])).tolist()[0]
            output_file.write(str(status) + '\n')

if __name__=='__main__':
    args = CommonArgParser().parse_args()
    local_map = construct_local_map(args)
    get_status(args,local_map)